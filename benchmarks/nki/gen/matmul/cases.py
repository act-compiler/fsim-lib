# The looped matmul cases, C[M, N] = A[M, K] @ B[K, N] in fp16.
# Each is named matmul/<M>x<K>x<N>, K the contracting dimension.
# Pure data: the runner imports it from the base env.
# The kernel's tiles need M and K multiples of 128, N of 512.

# Square sweep, M = K = N.
SQUARE = [512, 1024, 2048, 4096]

# Aspect ratios at 2048^3's work, as (M, K, N).
ASPECT = [
    (1024, 2048, 4096),
    (1024, 4096, 2048),
    (2048, 1024, 4096),
    (2048, 4096, 1024),
    (4096, 1024, 2048),
    (4096, 2048, 1024),
]

SEED = 0


def _case(m, k, n):
    return {
        'name': f'matmul/{m}x{k}x{n}',
        'dtype': 'float16',
        'seed': SEED,
        'M': m,
        'K': k,
        'N': n,
    }


CASES = [_case(s, s, s) for s in SQUARE] + [_case(*mkn) for mkn in ASPECT]
