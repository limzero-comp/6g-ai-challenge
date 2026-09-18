"""Score budget and finite-source-code checks; standard library, no training.

Read an existing Git revision without checking it out. Source-code accuracies
assume an explicitly described toy payload channel, NOT the competition link.
"""
import argparse
import itertools
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BITS = 1152


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def get_json(revision, path):
    return json.loads(git("show", revision + ":" + path))


def score_budget(metrics, target):
    efficiency = metrics["efficiency"]
    fairness = metrics["fairness_p10"]
    score = 0.7 * efficiency + 0.3 * fairness
    assert math.isclose(score, metrics["score"], abs_tol=1e-9)
    bins = metrics["snr_bins"]
    count = sum(v["count"] for v in bins.values())
    weighted = sum(v["count"] * v["efficiency"] for v in bins.values()) / count
    assert math.isclose(weighted, efficiency, abs_tol=2e-5)
    gains = {}
    for cutoff in (0, 5, 10, 15):
        delta_mean = sum(
            v["count"] * (100.0 - v["efficiency"])
            for label, v in bins.items()
            if int(label.split(",")[0][1:]) >= cutoff
        ) / count
        gains[str(cutoff)] = 0.7 * delta_mean
    return {
        "current": {"efficiency": efficiency, "fairness_p10": fairness, "score": score},
        "target": target,
        "gap": target - score,
        "required_mean_with_current_p10": (target - 0.3 * fairness) / 0.7,
        "required_mean_by_target_p10": {
            str(f): (target - 0.3 * f) / 0.7 for f in (50, 52, 54, 55, 56, 58, 60)
        },
        "gain_from_p10_50_holding_mean_fixed": 0.3 * (50 - fairness),
        "gain_from_p10_55_holding_mean_fixed": 0.3 * (55 - fairness),
        "gain_if_all_scores_above_snr_cutoff_become_100_holding_p10_fixed": gains,
        "caution": "The high-SNR replacements hold p10 fixed; they do not simulate a new global quantile.",
    }


def hamming74_encode(payload):
    """Four payload bits -> a systematic seven-bit Hamming codeword."""
    a, b, c, d = payload
    return (a ^ b ^ d, a ^ c ^ d, a, b ^ c ^ d, b, c, d)


def hamming74_quantize(source):
    """Seven source bits -> four-bit index of the nearest codeword.

    This is lossy source compression, not channel error correction. All 128
    source strings have a unique nearest codeword at distance zero or one.
    """
    syndrome = 0
    for position, bit in enumerate(source, 1):
        if bit:
            syndrome ^= position
    nearest = list(source)
    if syndrome:
        nearest[syndrome - 1] ^= 1
    payload = tuple(nearest[i] for i in (2, 4, 5, 6))
    assert hamming74_encode(payload) == tuple(nearest)
    return payload


def hamming74_analysis():
    sources = list(itertools.product((0, 1), repeat=7))
    payloads = list(itertools.product((0, 1), repeat=4))
    codewords = [hamming74_encode(p) for p in payloads]
    quantized = []
    distances = []
    for source in sources:
        payload = hamming74_quantize(source)
        decoded = hamming74_encode(payload)
        distance = sum(a != b for a, b in zip(source, decoded))
        all_distances = [sum(a != b for a, b in zip(source, c)) for c in codewords]
        assert distance == min(all_distances) and all_distances.count(distance) == 1
        quantized.append(payload)
        distances.append(distance)
    assert max(distances) == 1 and sum(distances) == 112
    accuracies = {}
    for p in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5):
        total = 0.0
        for source, payload in zip(sources, quantized):
            for flips in payloads:
                errors = sum(flips)
                probability = p ** errors * (1 - p) ** (4 - errors)
                received = tuple(a ^ b for a, b in zip(payload, flips))
                reconstruction = hamming74_encode(received)
                correct = sum(a == b for a, b in zip(source, reconstruction))
                total += probability * correct / (7 * len(sources))
        accuracies[str(p)] = {
            "source_accuracy_pct": 100 * total,
            "same_four_bit_budget_prefix_score_pct": 100 * (0.5 + 4 / 7 * (0.5 - p)),
        }
    assert math.isclose(accuracies["0.0"]["source_accuracy_pct"], 87.5)
    assert math.isclose(accuracies["0.5"]["source_accuracy_pct"], 50)
    return {
        "source_bits": 7, "payload_bits": 4,
        "enumerated_sources": len(sources), "codewords": len(codewords),
        "max_source_errors_if_payload_is_perfect": max(distances),
        "payload_bsc_checks": accuracies,
        "block_1152_with_four_direct_remainder_bits": {
            "payload_bits": 164 * 4 + 4,
            "mean_score_if_payload_is_perfect": (164 * 7 * 87.5 + 4 * 100) / BITS,
        },
        "caution": "Independent payload-bit BSC is a toy channel, not a measured wireless BER or score.",
    }


def majority_analysis(size):
    """Compress an odd-sized uniform binary block to its majority bit."""
    assert size % 2 == 1
    accuracy = sum(math.comb(size, ones) * max(ones, size - ones)
                   for ones in range(size + 1)) / (size * 2 ** size)
    groups, remainder = divmod(BITS, size)
    return {
        "source_bits": size, "payload_bits": 1,
        "accuracy_if_payload_is_perfect_pct": 100 * accuracy,
        "prefix_with_same_payload_budget_pct": 100 * (0.5 + 0.5 / size),
        "payload_bsc_checks": {
            str(p): 100 * (0.5 + (accuracy - 0.5) * (1 - 2 * p))
            for p in (0, 0.05, 0.1, 0.2, 0.5)
        },
        "block_1152_with_direct_remainder": {
            "payload_bits": groups + remainder,
            "mean_score_if_payload_is_perfect": 100 * (groups * size * accuracy + remainder) / BITS,
        },
    }


def association_check():
    """Prove the finite control protocol for every pair of 32 SNR bins.

    Control 0..30 separates bins <= control and > control. Control 31 means
    both UEs lie in the same bin and must use a separate symmetric fallback.
    No user index is given to the receiver. This only tests association;
    it does not implement a beam, fallback waveform, or wireless receiver.
    """
    collisions = 0
    checked = 0
    for q0, q1 in itertools.product(range(32), repeat=2):
        control = min(q0, q1) if q0 != q1 else 31
        assert control == (min(q1, q0) if q1 != q0 else 31)
        wire = tuple((control >> i) & 1 for i in range(5))
        assert sum(b << i for i, b in enumerate(wire)) == control
        if control == 31:
            assert q0 == q1
            collisions += 1
        else:
            ranks = (int(q0 > control), int(q1 > control))
            assert ranks == ((0, 1) if q0 < q1 else (1, 0))
            checked += 1
    return {"distinct_bin_pairs_verified": checked, "same_bin_fallback_pairs": collisions,
            "fallback_probability_if_independent_uniform_snr": collisions / 1024,
            "scope": "Finite protocol proof only; all same-bin cases require a real fallback."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/mac-mini-dev")
    parser.add_argument("--target", type=float, default=70.0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/gap_to_70.json")
    args = parser.parse_args()
    revision = git("rev-parse", "--verify", args.ref + "^{commit}")
    source_path = "reports/results/v3/eval_val.json"
    metrics = get_json(revision, source_path)
    result = {
        "provenance": {"revision": revision, "metrics_path": source_path,
                       "training_steps_executed": 0, "new_wireless_evaluations_executed": 0},
        "score_budget": score_budget(metrics, args.target),
        "formal_snr_bins": metrics["snr_bins"],
        "rate_policy_caps": {
            "v2_lowest_tier": {"snr": [-20, -15], "population_fraction": 0.125,
                               "bits": 16, "max_score": 50 + 50 * 16 / BITS},
            "v4_lowest_tier": {"snr": [-20, -14], "population_fraction": 0.15,
                               "bits": 96, "max_score": 50 + 50 * 96 / BITS},
            "scope": "Population p10 cannot exceed a tier cap if >10% of UE blocks are in that tier.",
        },
        "hamming74_source_compression": hamming74_analysis(),
        "majority_source_compression": [majority_analysis(n) for n in (3, 7, 15)],
        "shared_5bit_association_protocol": association_check(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "revision": revision,
                      "gap": result["score_budget"]["gap"],
                      "hamming74_ideal_accuracy": 87.5,
                      "association": result["shared_5bit_association_protocol"]}, indent=2))


if __name__ == "__main__":
    main()
