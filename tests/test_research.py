"""Contract/physics/gradient tests only. No optimizer step or model training."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from research.data import Channels, load_splits, make_splits
from research.evaluate import summarize
from research.link import (B_MAX, Link, load_design, normalize_downlink,
                           normalize_feedback, per_user_score)
from research.objective import objective
from analysis.audit_data import qam_table

ROOT = Path(__file__).resolve().parents[1]


def official_link(design):
    """Execute only the official class, not modelEval.py's loading/evaluation side effects."""
    tree = ast.parse((ROOT / "modelEval.py").read_text())
    kept = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MU_MIMO_Link"]
    namespace = dict(torch=torch, nn=torch.nn, DEVICE=torch.device("cpu"),
                     Encoder=design.Encoder, Transmitter=design.Transmitter, Receiver=design.Receiver,
                     NUM_UE=2, NUM_UPLINK_SUBCARRIERS=96, NUM_DOWNLINK_DATA_SUBCARRIERS=144,
                     NUM_DOWNLINK_CTRL_BITS=5, NUM_DOWNLINK_TX=16, NUM_MAX_BITS=B_MAX)
    exec(compile(ast.Module(body=kept, type_ignores=[]), "official_class", "exec"), namespace)
    return namespace["MU_MIMO_Link"]()


class DataAndScoreTests(unittest.TestCase):
    def test_mmap_npz_and_split_identity(self):
        rng = np.random.default_rng(7)
        shape = (20, 2, 2, 16, 144)
        re = rng.normal(size=shape).astype(np.float16)
        im = rng.normal(size=shape).astype(np.float16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.savez(path / "h.npz", real=re, imag=im)
            data = Channels(path / "h.npz")
            idx = [1, 4, 18]
            np.testing.assert_array_equal(data.take(idx), re[idx].astype(np.float32) + 1j * im[idx].astype(np.float32))
            parts = make_splits(20)
            metadata = {"sampled_sha256": data.fingerprint()}
            np.savez(path / "splits.npz", **parts, metadata=json.dumps(metadata))
            loaded, _ = load_splits(path / "splits.npz", data)
            self.assertEqual([len(loaded[k]) for k in ("train", "val", "test")], [16, 2, 2])
            np.savez_compressed(path / "compressed.npz", real=re, imag=im)
            with self.assertRaisesRegex(ValueError, "uncompressed"):
                Channels(path / "compressed.npz")

    def test_group_disjointness(self):
        groups = np.repeat(np.arange(20), 3)
        parts = make_splits(len(groups), groups=groups)
        sets = [set(groups[v]) for v in parts.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_exact_score_and_zero_llr_semantics(self):
        bits = torch.zeros(3, B_MAX)
        logits = -torch.ones(3, B_MAX)
        lengths = torch.tensor([B_MAX, 144, 1])
        expected = 50.0 + 50.0 * lengths / B_MAX
        torch.testing.assert_close(per_user_score(bits, logits, lengths), expected)
        zero_logits = torch.zeros(3, B_MAX)
        # The official >=0 decision chooses one: zeros are wrong for target zeros.
        torch.testing.assert_close(per_user_score(bits, zero_logits, lengths), 50.0 - 50.0 * lengths / B_MAX)
        self.assertEqual(summarize([0, 100])["fairness_p10"], 10.0)

    def test_normalization(self):
        u = normalize_feedback(torch.randn(3, 96, dtype=torch.complex64))
        x = normalize_downlink(torch.randn(3, 16, 144, dtype=torch.complex64))
        torch.testing.assert_close(u.abs().square().mean(1), torch.ones(3))
        torch.testing.assert_close(x.abs().square().sum(1).mean(1), torch.ones(3))
        with self.assertRaises(ValueError):
            normalize_downlink(torch.zeros_like(x))

    def test_qam_inverse_and_high_snr(self):
        dbs, tables, checks = qam_table(np.random.default_rng(0), mc=2000)
        for check in checks.values():
            self.assertEqual(check["noiseless_errors"], 0)
            self.assertAlmostEqual(check["average_constellation_power"], 1.0)
        self.assertEqual(dbs[-1], 40)
        self.assertTrue(all(curve[-1] == 0 for curve in tables.values()))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_candidates(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                self.check_candidate(version)

    def check_candidate(self, version):
        torch.manual_seed(42)
        source = ROOT / "models" / ("modelDesign_v%d.py" % version)
        design = load_design(source)
        link = Link(design).eval()
        h = torch.randn(3, 2, 2, 16, 144, dtype=torch.complex64)
        snr = torch.tensor([[-20.0, 0.0, 20.0], [20.0, -20.0, 0.0]])
        bits = [torch.randint(0, 2, (3, B_MAX)).float() for _ in range(2)]
        before = [p.detach().clone() for p in link.parameters()]
        logits, lengths, aux = link(h, bits, snr, collect_aux=True)
        self.assertEqual(aux["h_hat"].shape, h.shape)
        loss, _ = objective(bits, logits, lengths, score_weight=0.1, tail_weight=0.03)
        loss.backward()
        for name in ("encoder", "transmitter", "receiver"):
            grads = [p.grad for p in getattr(link, name).parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0, name)
        self.assertTrue(all(torch.equal(old, current) for old, current in zip(before, link.parameters())))
        link.zero_grad(set_to_none=True)

        # The official class really instantiates and calls the zero-argument models.
        official = official_link(design).eval()
        official._encoder.load_state_dict(link.encoder.state_dict())
        official._transmitter.load_state_dict(link.transmitter.state_dict())
        official._receiver.load_state_dict(link.receiver.state_dict())
        with torch.no_grad():
            for j in range(3):
                one_h, one_snr = h[j:j + 1], snr[:, j:j + 1]
                one_bits = [b[j:j + 1] for b in bits]
                torch.manual_seed(11)
                custom, count, _ = link(one_h, one_bits, one_snr)
                torch.manual_seed(11)
                reference = official([one_h[:, i] for i in range(2)], one_bits, one_snr, one_snr - 10)
                for ue in range(2):
                    n = int(count[ue].item())
                    self.assertEqual(reference[ue].shape, (1, n))
                    torch.testing.assert_close(custom[ue][:, :n], reference[ue], rtol=3e-4, atol=3e-4)
                    direct = 100 * (((reference[ue] >= 0) == one_bits[ue][:, :n].bool()).sum(1)
                                    + (B_MAX - n) * 0.5) / B_MAX
                    torch.testing.assert_close(per_user_score(one_bits[ue], custom[ue], count[ue]), direct)

            # Vary UE ordering; no hidden user ID or invocation-state cache is permitted.
            feedback = [normalize_feedback(link.encoder(h[:, i], snr[i])) for i in range(2)]
            x, ctrl = link.transmitter(bits, feedback, snr)
            swapped, swapped_ctrl = link.transmitter(bits[::-1], feedback[::-1], snr.flip(0))
            torch.testing.assert_close(x, swapped, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(ctrl, swapped_ctrl)
            mode = (ctrl.long() * (2 ** torch.arange(5))).sum(1)
            expected_mode = torch.floor((snr.min(0).values + 20) * 0.8).long().clamp(0, 31)
            torch.testing.assert_close(mode, expected_mode)
            # Every RE/bit decision must be per sample, not batch statistics.
            y = torch.randn(3, 2, 144, dtype=torch.complex64)
            full = link.receiver.forward_full(y, h[:, 0], ctrl, snr[0])
            for j in range(3):
                single = link.receiver.forward_full(y[j:j + 1], h[j:j + 1, 0], ctrl[j:j + 1], snr[0, j:j + 1])
                torch.testing.assert_close(full[j:j + 1], single, rtol=1e-4, atol=1e-4)
            # Zero channels still have valid nonzero feedback/transmit energy and finite outputs.
            link(torch.zeros_like(h), bits, snr)

            if version != 1:
                # Test the block coder itself (before normalization): one input
                # bit must affect many other REs, not just rescale their energy.
                mod = link.transmitter._modulator
                source_bits = bits[0][:1].clone()
                full_length = torch.tensor([B_MAX])
                z_before = mod(source_bits, snr[0, :1], full_length)
                source_bits[0, 0] = 1 - source_bits[0, 0]
                z_after = mod(source_bits, snr[0, :1], full_length)
                affected = ((z_before - z_after).abs().sum(-1) > 1e-7).sum()
                self.assertGreater(int(affected), 100)

        if version == 2:
            with self.assertRaises(ValueError):
                link.receiver(y, h[:, 0], ctrl, snr[0])
            # A discarded suffix must not change the transmitted waveform.
            changed = [b.clone() for b in bits]
            for ue in range(2):
                valid = link.receiver.valid_lengths(snr[ue], ctrl)
                mask = torch.arange(B_MAX)[None] >= valid[:, None]
                changed[ue][mask] = 1 - changed[ue][mask]
            with torch.no_grad():
                changed_x, _ = link.transmitter(changed, feedback, snr)
            torch.testing.assert_close(x, changed_x)

        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "modelDesign.py"
            copied.write_bytes(source.read_bytes())
            standalone = Link(load_design(copied)).eval()
            for name in ("encoder", "transmitter", "receiver"):
                file = Path(tmp) / (name + ".pth")
                torch.save(getattr(link, name).state_dict(), file)
                getattr(standalone, name).load_state_dict(torch.load(file, weights_only=True), strict=True)
            self.assertEqual(sum(p.numel() for p in standalone.parameters()),
                             sum(p.numel() for p in link.parameters()))
        print("v%d: contracts, official parity, gradients without update, symmetry, serialization OK (%d parameters)"
              % (version, sum(p.numel() for p in link.parameters())), flush=True)


if __name__ == "__main__":
    unittest.main()
