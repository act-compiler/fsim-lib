# Per-kernel builders: the NKI kernel and its seeded inputs.
# A builder takes a case and a Rand and returns a Build.
# Each kernel breaks one trn2 memory limit at an explicit address.
from collections.abc import Callable
from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

F32 = nl.float32
# trn2's bytes per partition.
SBUF = 229376
PSUM = 16384


@dataclass
class Build:
    kernel: Callable
    tensors: dict
    kwargs: dict


class Rand:
    """Seeded numpy draws: small integers as fp32, exact in any sum."""

    def __init__(self, seed):
        self.g = np.random.default_rng(seed)

    def ints(self, shape):
        return self.g.integers(-2, 3, shape).astype(np.float32)


def copy_through(x, t):
    # HBM to the tile and back.
    out = nl.ndarray(x.shape, x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    return out


def matmul(w, x, dst):
    # w.T @ x into dst, then out through SBUF.
    out = nl.ndarray((128, 512), F32, buffer=nl.shared_hbm)
    ws = nl.ndarray((128, 128), F32, buffer=nl.sbuf, address=(0, 0))
    xs = nl.ndarray((128, 512), F32, buffer=nl.sbuf, address=(0, 512))
    s = nl.ndarray((128, 512), F32, buffer=nl.sbuf, address=(0, 2560))
    nisa.dma_copy(dst=ws, src=w)
    nisa.dma_copy(dst=xs, src=x)
    nisa.nc_matmul(dst=dst, stationary=ws, moving=xs, accumulate=False)
    nisa.activation(dst=s, op=nl.copy, data=dst)
    nisa.dma_copy(dst=out, src=s)
    return out


@nki.jit
def sbuf_capacity(x):
    # 512 B per partition from 229120, past SBUF's 229376.
    return copy_through(x, nl.ndarray((128, 128), F32, buffer=nl.sbuf, address=(0, SBUF - 256)))


@nki.jit
def partitions(x):
    # 128 partitions from partition 64, past 128.
    return copy_through(x, nl.ndarray((128, 128), F32, buffer=nl.sbuf, address=(64, 0)))


@nki.jit
def psum_capacity(w, x):
    # 4 KiB per partition from 14336, past PSUM's 16384.
    p = nl.ndarray((128, 1024), F32, buffer=nl.psum, address=(0, PSUM - 2048))
    return matmul(w, x, p[:, 0:512])


@nki.jit
def psum_bank(w, x):
    # A 2 KiB dst from byte 1024 crosses into bank 1.
    p = nl.ndarray((128, 1024), F32, buffer=nl.psum, address=(0, 0))
    return matmul(w, x, p[:, 256:768])


def tile_build(kernel):
    return lambda c, r: Build(kernel, {'x': r.ints((128, 128))}, {})


def matmul_build(kernel):
    return lambda c, r: Build(kernel, {'w': r.ints((128, 128)), 'x': r.ints((128, 512))}, {})


BUILDERS = {
    'sbuf_capacity': tile_build(sbuf_capacity),
    'partitions': tile_build(partitions),
    'psum_capacity': matmul_build(psum_capacity),
    'psum_bank': matmul_build(psum_bank),
}
