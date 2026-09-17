"""Read the official NPZ without a multi-GB complex64 copy; split channel rows first."""
import hashlib
import json
import struct
import zipfile
from pathlib import Path

import numpy as np


def _stored_member(path, member):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError("Use an uncompressed NPZ or a directory with real.npy/imag.npy. "
                             "Extract compressed NPZ members once before running experiments.")
        with open(path, "rb") as raw:
            raw.seek(info.header_offset)
            header = raw.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            name_len, extra_len = struct.unpack_from("<HH", header, 26)
            start = info.header_offset + 30 + name_len + extra_len
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Unsupported NPY format: %s" % (version,))
            offset = start + stream.tell()
        if dtype.hasobject:
            raise ValueError("Object arrays are forbidden")
        expected = int(np.prod(shape)) * dtype.itemsize
        if offset + expected > start + info.file_size:
            raise ValueError("Truncated array")
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape,
                     order="F" if fortran else "C")


class Channels:
    def __init__(self, path):
        path = Path(path)
        if path.is_dir():
            self.real = np.load(path / "real.npy", mmap_mode="r", allow_pickle=False)
            self.imag = np.load(path / "imag.npy", mmap_mode="r", allow_pickle=False)
        else:
            self.real = _stored_member(path, "real.npy")
            self.imag = _stored_member(path, "imag.npy")
        if self.real.shape != self.imag.shape or self.real.shape[1:] != (2, 2, 16, 144):
            raise ValueError("Expected matching [N,2,2,16,144] real/imag arrays")
        self.path = path

    def __len__(self):
        return len(self.real)

    def take(self, indices):
        real = np.asarray(self.real[indices], dtype=np.float32)
        imag = np.asarray(self.imag[indices], dtype=np.float32)
        result = real + np.complex64(1j) * imag
        if not np.isfinite(result).all():
            raise ValueError("Non-finite channel values")
        return result

    def fingerprint(self):
        # This is a sampled identity check, explicitly not a full archive checksum.
        rows = np.unique(np.linspace(0, len(self) - 1, min(32, len(self))).astype(int))
        digest = hashlib.sha256()
        digest.update(str((self.real.shape, self.real.dtype.str, self.imag.dtype.str)).encode())
        for array in (self.real, self.imag):
            digest.update(np.ascontiguousarray(array[rows]).tobytes())
        return digest.hexdigest()


def make_splits(n, seed=20260918, mode="random", groups=None):
    if n < 10:
        raise ValueError("Need at least 10 channel samples")
    rng = np.random.default_rng(seed)
    units = np.arange(n) if groups is None else np.unique(groups)
    if groups is not None and len(groups) != n:
        raise ValueError("Group ID count must match channel rows")
    if len(units) < 10:
        raise ValueError("Need at least 10 independent rows/groups")
    if mode == "random":
        units = rng.permutation(units)
    elif mode != "ordered":
        raise ValueError("Unknown split mode")
    pieces = np.split(units, [int(0.8 * len(units)), int(0.9 * len(units))])
    if groups is not None:
        pieces = [np.flatnonzero(np.isin(groups, part)) for part in pieces]
    return dict(zip(("train", "val", "test"), pieces))


def load_splits(path, channels):
    with np.load(path, allow_pickle=False) as archive:
        parts = {name: archive[name].copy() for name in ("train", "val", "test")}
        metadata = json.loads(str(archive["metadata"]))
    flat = np.concatenate(list(parts.values()))
    if len(flat) != len(channels) or len(np.unique(flat)) != len(channels):
        raise ValueError("Splits do not form a disjoint partition")
    if flat.min() != 0 or flat.max() != len(channels) - 1:
        raise ValueError("Invalid split indices")
    if metadata["sampled_sha256"] != channels.fingerprint():
        raise ValueError("Dataset sampled fingerprint mismatch")
    return parts, metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--mode", choices=["random", "ordered"], default="random")
    parser.add_argument("--groups", help="Optional NPY with scenario/group ID for every row")
    args = parser.parse_args()
    data = Channels(args.data)
    groups = np.load(args.groups, allow_pickle=False) if args.groups else None
    parts = make_splits(len(data), args.seed, args.mode, groups)
    metadata = {"seed": args.seed, "mode": args.mode, "groups": args.groups,
                "sampled_sha256": data.fingerprint(), "n": len(data),
                "warning": "Sampled fingerprint, not a complete file hash; no scenario IDs supplied"
                if groups is None else "Group-disjoint split"}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing split")
    with destination.open("wb") as stream:
        np.savez(stream, **parts, metadata=json.dumps(metadata))
    print(json.dumps({**metadata, "sizes": {k: len(v) for k, v in parts.items()}}, indent=2))


if __name__ == "__main__":
    main()
