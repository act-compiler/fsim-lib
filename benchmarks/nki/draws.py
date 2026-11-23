# Each simulated sample's output and its input set's golden, by <case>/out/draws.bin.
# draws.bin holds each sample's input-set index, a little-endian uint32 each.
# out/<name>.bin holds every sample's output back to back, golden/<name>.bin every input set's.
import os

import numpy as np


# An out/ that does not fit draws.bin or the case's golden files.
class Mismatch(Exception):
    pass


def read(folder):
    # Each sample's input-set index.
    path = os.path.join(folder, 'out', 'draws.bin')
    if not os.path.isfile(path):
        raise Mismatch('no out/draws.bin')
    indices = np.fromfile(path, '<u4')
    if not indices.size:
        raise Mismatch('out/draws.bin is empty')
    return indices


def pairs(folder, name, indices, kind):
    # Yields (where, golden, output) per sample, as float64.
    golden = os.path.join(folder, 'golden', name)
    out = os.path.join(folder, 'out', name)
    if not os.path.isfile(out):
        raise Mismatch(f'no out/{name}')
    item = np.dtype(kind).itemsize
    size, want = os.path.getsize(out), os.path.getsize(golden)
    one = size // indices.size
    if not one or size % indices.size or one % item or want % one:
        raise Mismatch(f'out/{name} is {size} B for {indices.size} samples, golden {want} B')
    if indices.max() >= want // one:
        raise Mismatch(f'out/draws.bin draws input set {indices.max()} of {want // one}')
    for k, s in enumerate(indices):
        where = f'out/{name}' if indices.size == 1 else f'out/{name} sample {k}'
        ref = np.fromfile(golden, kind, one // item, offset=int(s) * one).astype(np.float64)
        act = np.fromfile(out, kind, one // item, offset=k * one).astype(np.float64)
        yield where, ref, act
