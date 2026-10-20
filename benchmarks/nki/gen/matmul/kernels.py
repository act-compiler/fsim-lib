# The matmul kernel, and the Build that gen.py compiles and records.
from collections.abc import Callable
from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl

F16 = nl.float16
F32 = nl.float32
# NKI's matmul tutorial tiles: M, K by partitions, N by a bank.
TILE_M = nl.tile_size.gemm_stationary_fmax
TILE_K = nl.tile_size.pmax
TILE_N = nl.tile_size.gemm_moving_fmax
# Partitions, the identity's size.
P = TILE_K


@dataclass
class Build:
    kernel: Callable
    tensors: dict
    kwargs: dict


@nki.jit
def matmul(tensor_a, tensor_b, tensor_id):
    # tensor_c by tiles of TILE_M rows and TILE_N columns.
    m, k = tensor_a.shape
    n = tensor_b.shape[1]
    kt = k // TILE_K
    tensor_c = nl.ndarray((m, n), F16, buffer=nl.shared_hbm)
    ident = nl.ndarray((P, P), F16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ident, src=tensor_id)
    # tensor_b as [row in k tile, k tile, column].
    b_tiles = tensor_b.reshape_dim(0, (kt, TILE_K)).permute((1, 0, 2))
    # Transposes per PSUM bank.
    per = TILE_N // TILE_M
    for i in range(m // TILE_M):
        a = nl.ndarray((TILE_M, k), F16, buffer=nl.sbuf)
        nisa.dma_copy(dst=a, src=tensor_a[i * TILE_M : (i + 1) * TILE_M, :])
        # A's k tiles transposed against the identity, a bank at a time.
        at = nl.ndarray((TILE_K, kt * TILE_M), F16, buffer=nl.sbuf)
        for g in range(0, kt, per):
            # The last bank may hold fewer transposes.
            w = min(per, kt - g)
            t = nl.ndarray((TILE_K, w * TILE_M), F32, buffer=nl.psum)
            for j in range(w):
                x = g + j
                nisa.nc_matmul(
                    dst=t[:, j * TILE_M : (j + 1) * TILE_M],
                    stationary=a[:, x * TILE_K : (x + 1) * TILE_K],
                    moving=ident,
                    accumulate=False,
                )
            nisa.activation(dst=at[:, g * TILE_M : (g + w) * TILE_M], op=nl.copy, data=t)
        for j in range(n // TILE_N):
            b = nl.ndarray((TILE_K, kt, TILE_N), F16, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=b_tiles[:, :, j * TILE_N : (j + 1) * TILE_N])
            # Accumulates over k tiles in one PSUM bank.
            acc = nl.ndarray((TILE_M, TILE_N), F32, buffer=nl.psum)
            for x in range(kt):
                nisa.nc_matmul(
                    dst=acc,
                    stationary=at[:, x * TILE_M : (x + 1) * TILE_M],
                    moving=b[:, x, :],
                    accumulate=x > 0,
                )
            c = nl.ndarray((TILE_M, TILE_N), F16, buffer=nl.sbuf)
            nisa.activation(dst=c, op=nl.copy, data=acc)
            nisa.dma_copy(
                dst=tensor_c[i * TILE_M : (i + 1) * TILE_M, j * TILE_N : (j + 1) * TILE_N], src=c
            )
    return tensor_c
