# The nkilib cases of the nki suite, as pure data.
# Each case is generated into benchmarks/nki/<name>/ by gen.py.
# A case is one core's tensor-parallel shard of its model, bf16;
# sweeps vary one knob each around the Llama 3 8B base.

# Models, each with its TP degree.
LLAMA3_8B = {
    'model': 'llama3_8b',
    'tp': 4,
    'hidden': 4096,
    'intermediate': 14336,
    'q_heads': 32,
    'kv_heads': 8,
    'head_dim': 128,
}
LLAMA3_70B = {
    'model': 'llama3_70b',
    'tp': 8,
    'hidden': 8192,
    'intermediate': 28672,
    'q_heads': 64,
    'kv_heads': 8,
    'head_dim': 128,
}

# MoE: Llama 4 Scout's routed experts, since Llama 3 8B has no MoE.
MOE = {
    'model': 'llama4_scout',
    'tp': 8,
    'hidden': 5120,
    'intermediate': 8192,
    'experts': 16,
    'top_k': 1,
}

# Base serving point: one request, prefill then decode.
CTE_SEQ = 2048
TKG_BATCH = 1
TKG_KV = 2048

# Sweep points beside the base; mlp_tkg needs B*S <= 96.
CTE_SEQ_SWEEP = [512, 1024, 4096, 8192]
TKG_BATCH_SWEEP = [4, 16, 64]
TKG_KV_SWEEP = [512, 1024, 4096, 8192]

CTE_KERNELS = ['attention_cte', 'mlp_cte', 'qkv_cte', 'output_proj_cte']
TKG_KERNELS = ['attention_tkg', 'mlp_tkg', 'qkv_tkg', 'output_proj_tkg']

# Shape parameters a folder name may spell, in this order.
NAME_ORDER = ['seq', 'batch', 'kv']


def shard(m):
    # TP splits heads, KV heads and intermediate; hidden stays whole.
    tp = m['tp']
    s = {**m, 'intermediate': m['intermediate'] // tp}
    if 'q_heads' in m:
        s['q_heads'] = m['q_heads'] // tp
        s['kv_heads'] = max(1, m['kv_heads'] // tp)
    return s


def case(kernel, model, dtype='bfloat16', sharded=True, folder=None, **params):
    return {
        'folder': f'nkilib/{folder or kernel}',
        'dtype': dtype,
        'seed': 0,
        'kernel': kernel,
        **(shard(model) if sharded else model),
        **params,
    }


def cte(kernel, model, seq=CTE_SEQ):
    return case(kernel, model, batch=1, seq=seq)


def tkg(kernel, model, batch=TKG_BATCH, kv=TKG_KV):
    # Only attention reads a KV cache.
    kw = {'kv': kv} if kernel == 'attention_tkg' else {}
    return case(kernel, model, batch=batch, seq=1, **kw)


def name_all(cases):
    # A name spells model, tp and the parameters that vary within its folder.
    for c in cases:
        peers = [p for p in cases if p['folder'] == c['folder']]
        vary = [k for k in NAME_ORDER if len({p.get(k) for p in peers}) > 1]
        c['name'] = f'{c["folder"]}/{c["model"]}_tp{c["tp"]}' + ''.join(f'_{k}{c[k]}' for k in vary)
    return cases


CASES = []

# CTE/TKG pairs: base, sweeps and 70B.
for k in CTE_KERNELS:
    CASES += [cte(k, LLAMA3_8B, s) for s in sorted([CTE_SEQ] + CTE_SEQ_SWEEP)]
    CASES.append(cte(k, LLAMA3_70B))
for k in TKG_KERNELS:
    CASES += [tkg(k, LLAMA3_8B, b) for b in sorted([TKG_BATCH] + TKG_BATCH_SWEEP)]
    if k == 'attention_tkg':
        CASES += [tkg(k, LLAMA3_8B, kv=v) for v in TKG_KV_SWEEP]
    CASES.append(tkg(k, LLAMA3_70B))

# Norms, RoPE and cumsum keep their full width.
for m in [LLAMA3_8B, LLAMA3_70B]:
    for k in ['rmsnorm_tkg', 'layernorm_tkg']:
        CASES.append(case(k, m, sharded=False, batch=TKG_BATCH, seq=1))

# RoPE over the query heads of a prefill.
CASES.append(case('rope', LLAMA3_8B, sharded=False, batch=1, seq=CTE_SEQ, rope_theta=500000.0))

# Cumsum over prefill rows of the hidden width, fp32 accumulation.
CASES.append(case('cumsum', LLAMA3_8B, sharded=False, dtype='float32', seq=CTE_SEQ))

# MoE decode, data-dependent.
for mode in ['selective', 'all_expert']:
    CASES.append(
        case(
            'moe_tkg',
            MOE,
            folder=f'moe_tkg/{mode}',
            batch=TKG_BATCH,
            seq=1,
            mode=mode,
        )
    )

name_all(CASES)
