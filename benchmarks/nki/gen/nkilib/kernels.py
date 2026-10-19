# Per-kernel builders: the NKI kernel, its seeded inputs and its torch reference.
# A builder takes a case and a Rand and returns a Build.
# Kernels that take their output buffer get a thin jit wrapper here.
import math
from collections.abc import Callable
from dataclasses import dataclass

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.attention.attention_cte_torch import attention_cte_torch_ref
from nkilib.core.attention.attention_tkg import attention_tkg
from nkilib.core.attention.attention_tkg_torch import attention_tkg_torch_ref
from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
from nkilib.core.cumsum.cumsum import cumsum
from nkilib.core.cumsum.cumsum_torch import cumsum_torch_ref
from nkilib.core.embeddings.rope import RoPE
from nkilib.core.embeddings.rope_torch import RoPE_torch_ref
from nkilib.core.mlp.mlp import mlp
from nkilib.core.mlp.mlp_torch import mlp_torch_ref
from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.moe.moe_tkg.moe_tkg_torch import moe_tkg_torch_ref
from nkilib.core.output_projection.output_projection_cte.output_projection_cte import (
    output_projection_cte,
)
from nkilib.core.output_projection.output_projection_cte.output_projection_cte_torch import (
    output_projection_cte_torch_ref,
)
from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.output_projection.output_projection_tkg_torch import (
    output_projection_tkg_torch_ref,
)
from nkilib.core.qkv.qkv import qkv
from nkilib.core.qkv.qkv_torch import qkv_torch_ref
from nkilib.core.subkernels.layernorm_tkg import layernorm_tkg
from nkilib.core.subkernels.layernorm_torch import layernorm_tkg_torch_ref_lnc1
from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg
from nkilib.core.subkernels.rmsnorm_torch import rmsnorm_tkg_torch_ref_lnc1
from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.common_types import ExpertAffinityScaleMode, NormType
from nkilib.core.utils.logging import get_logger


@dataclass
class Build:
    kernel: Callable
    tensors: dict
    kwargs: dict
    # ref(t) gets the tensors as fp32 torch and returns the golden.
    ref: Callable


class Rand:
    """Seeded torch draws: normal as numpy in the kernel's dtype, uniform as fp32 torch."""

    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)

    def normal(self, shape, scale=1.0, dtype='bfloat16', mean=0.0):
        return to_numpy(torch.randn(shape, generator=self.g) * scale + mean, dtype)

    def uniform(self, shape):
        return torch.rand(shape, generator=self.g)


def to_numpy(t, dtype):
    if dtype == 'bfloat16':
        return t.to(torch.bfloat16).view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy().astype(dtype)


def to_torch(a):
    return torch.from_numpy(a if a.dtype.kind in 'iu' else a.astype(np.float32))


def weight(r, fan_in, shape):
    return r.normal(shape, 1.0 / math.sqrt(fan_in))


def single(res):
    # The one output of a ref, from a tensor, dict or tuple.
    if isinstance(res, dict):
        res = res['out'] if 'out' in res else next(v for v in res.values() if v is not None)
    if isinstance(res, (tuple, list)):
        res = res[0]
    return res


# Attention


def attention_cte_build(c, r):
    b, s, d = c['batch'], c['seq'], c['head_dim']
    bq, bkv = b * c['q_heads'], b * c['kv_heads']
    t = {
        'q': r.normal((bq, s, d)),
        'k': r.normal((bkv, d, s)),
        'v': r.normal((bkv, s, d)),
    }
    kw = {'scale': 1.0 / math.sqrt(d), 'causal_mask': True}
    return Build(attention_cte, t, kw, lambda x: attention_cte_torch_ref(**x, **kw))


def tkg_config(bs, q_head, s_active, s_prior, d):
    # A full prior cache, Q and K in SBUF.
    return AttnTKGConfig(
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=s_prior,
        full_sprior=s_prior,
        d_head=d,
        qk_in_sb=True,
    )


@nki.jit
def attention_tkg_kernel(q, k_active, v_active, k_prior, v_prior, mask):
    # Without fused RoPE the kernel takes Q and K from SBUF.
    bs, _, s_prior, d = v_prior.shape
    s_active = v_active.shape[2]
    q_head = q.shape[1] // (bs * s_active)
    cfg = tkg_config(bs, q_head, s_active, s_prior, d)
    sbm = create_auto_alloc_manager(logger=get_logger('attention_tkg'))
    sbm.open_scope(name='attention_tkg')
    q_sb = sbm.alloc_stack(q.shape, dtype=q.dtype, buffer=nl.sbuf)
    k_sb = sbm.alloc_stack(k_active.shape, dtype=k_active.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k_active)
    out = nl.ndarray((bs, q_head, d, s_active), dtype=q.dtype, buffer=nl.shared_hbm, name='out')
    attention_tkg(q_sb, k_sb, v_active, k_prior, v_prior, mask, out, cfg, sbm)
    sbm.close_scope()
    return out


def attention_tkg_build(c, r):
    # GQA folds the KV heads into the batch, one KV head each.
    bs, qh = c['batch'] * c['kv_heads'], c['q_heads'] // c['kv_heads']
    d, sa, sp = c['head_dim'], c['seq'], c['kv']
    t = {
        'q': r.normal((d, bs * qh * sa), 1.0 / math.sqrt(d)),
        'k_active': r.normal((d, bs * sa)),
        'v_active': r.normal((bs, 1, sa, d)),
        'k_prior': r.normal((bs, 1, d, sp)),
        'v_prior': r.normal((bs, 1, sp, d)),
    }
    # Each sequence holds a seeded context length; the active tokens always attend.
    lens = (r.uniform((c['batch'],)) * (sp - sa)).long() + 1
    pos = torch.arange(sp)[:, None]
    valid = (pos < lens[None, :]) | (pos >= sp - sa)
    valid = valid.repeat_interleave(c['kv_heads'], dim=1)
    t['mask'] = valid[:, :, None, None].expand(sp, bs, qh, sa).to(torch.uint8).numpy().copy()
    cfg = tkg_config(bs, qh, sa, sp, d)

    def ref(x):
        out = torch.zeros((bs, qh, d, sa))
        return attention_tkg_torch_ref[1](**x, out=out, cfg=cfg, sbm=None)

    return Build(attention_tkg_kernel, t, {}, ref)


# MLP, QKV and output projection


def mlp_build(c, r):
    h, i = c['hidden'], c['intermediate']
    t = {
        'hidden_tensor': r.normal((c['batch'], c['seq'], h)),
        'gate_proj_weights_tensor': weight(r, h, (h, i)),
        'up_proj_weights_tensor': weight(r, h, (h, i)),
        'down_proj_weights_tensor': weight(r, i, (i, h)),
    }
    return Build(mlp, t, {}, lambda x: mlp_torch_ref[1](**x))


def qkv_build(c, r):
    h = c['hidden']
    i = (c['q_heads'] + 2 * c['kv_heads']) * c['head_dim']
    t = {
        'input': r.normal((c['batch'], c['seq'], h)),
        'fused_qkv_weights': weight(r, h, (h, i)),
        'gamma_norm_weights': r.normal((1, h), 0.1, mean=1.0),
    }
    kw = {'fused_norm_type': NormType.RMS_NORM}
    return Build(qkv, t, kw, lambda x: qkv_torch_ref(**x, **kw))


def output_proj_cte_build(c, r):
    n, d, h = c['q_heads'], c['head_dim'], c['hidden']
    t = {
        'attention': r.normal((c['batch'], n, d, c['seq'])),
        'weight': weight(r, n * d, (n * d, h)),
    }
    return Build(output_projection_cte, t, {}, lambda x: output_projection_cte_torch_ref(**x))


def output_proj_tkg_build(c, r):
    n, d, h = c['q_heads'], c['head_dim'], c['hidden']
    t = {
        'attention': r.normal((d, c['batch'], n, c['seq'])),
        'weight': weight(r, n * d, (n * d, h)),
    }
    return Build(output_projection_tkg, t, {}, lambda x: output_projection_tkg_torch_ref(**x))


# Norms


@nki.jit
def rmsnorm_tkg_kernel(input, gamma):
    b, s, h = input.shape
    out = nl.ndarray((128, b * s, h // 128), dtype=input.dtype, buffer=nl.shared_hbm, name='out')
    rmsnorm_tkg(input, gamma, out)
    return out


@nki.jit
def layernorm_tkg_kernel(input, gamma, beta):
    b, s, h = input.shape
    out = nl.ndarray((128, b * s, h // 128), dtype=input.dtype, buffer=nl.shared_hbm, name='out')
    layernorm_tkg(input, gamma, out, beta)
    return out


def norm_inputs(c, r, beta):
    h = c['hidden']
    t = {
        'input': r.normal((c['batch'], c['seq'], h)),
        'gamma': r.normal((1, h), 0.1, mean=1.0),
    }
    if beta:
        t['beta'] = r.normal((1, h), 0.1)
    return t


def rmsnorm_tkg_build(c, r):
    return Build(
        rmsnorm_tkg_kernel,
        norm_inputs(c, r, False),
        {},
        lambda x: rmsnorm_tkg_torch_ref_lnc1(**x, output=None),
    )


def layernorm_tkg_build(c, r):
    return Build(
        layernorm_tkg_kernel,
        norm_inputs(c, r, True),
        {},
        lambda x: layernorm_tkg_torch_ref_lnc1(**x, output=None),
    )


# RoPE and cumsum


def rope_build(c, r):
    d, b, n, s = c['head_dim'], c['batch'], c['q_heads'], c['seq']
    # Llama 3 frequencies at positions 0..S-1.
    inv = 1.0 / (c['rope_theta'] ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = inv[:, None] * torch.arange(s, dtype=torch.float64)[None, :]
    ang = ang[:, None, :].expand(d // 2, b, s).float()
    t = {
        'x_in': r.normal((d, b, n, s)),
        'cos': to_numpy(torch.cos(ang), 'bfloat16'),
        'sin': to_numpy(torch.sin(ang), 'bfloat16'),
    }
    return Build(RoPE, t, {}, lambda x: RoPE_torch_ref(**x))


def cumsum_build(c, r):
    t = {'x': r.normal((c['seq'], c['hidden']), dtype=np.float32)}
    return Build(cumsum, t, {}, lambda x: cumsum_torch_ref(**x))


# MoE


def moe_tkg_build(c, r):
    tk, h, i, e, k = c['batch'] * c['seq'], c['hidden'], c['intermediate'], c['experts'], c['top_k']
    t = {
        'hidden_input': r.normal((tk, h)),
        'expert_gate_up_weights': weight(r, h, (e, h, 2, i)),
        'expert_down_weights': weight(r, i, (e, i, h)),
    }
    # Router: top-k of seeded logits, sigmoid scores as in Llama 4.
    logits = torch.randn((tk, e), generator=r.g)
    top, idx = torch.topk(logits, k, dim=-1)
    aff = torch.zeros_like(logits).scatter(1, idx, torch.sigmoid(top))
    t['expert_affinities'] = to_numpy(aff, np.float32)
    t['expert_index'] = idx.to(torch.int32).numpy()
    all_expert = c['mode'] == 'all_expert'
    kw = {
        'is_all_expert': all_expert,
        'expert_affinities_scaling_mode': ExpertAffinityScaleMode.POST_SCALE,
    }
    if all_expert:
        t['rank_id'] = np.zeros((1, 1), dtype=np.int32)
        kw['mask_unselected_experts'] = True
    return Build(moe_tkg, t, kw, lambda x: moe_tkg_torch_ref(**x, **kw))


BUILDERS = {
    'attention_cte': attention_cte_build,
    'attention_tkg': attention_tkg_build,
    'mlp_cte': mlp_build,
    'mlp_tkg': mlp_build,
    'qkv_cte': qkv_build,
    'qkv_tkg': qkv_build,
    'output_proj_cte': output_proj_cte_build,
    'output_proj_tkg': output_proj_tkg_build,
    'rmsnorm_tkg': rmsnorm_tkg_build,
    'layernorm_tkg': layernorm_tkg_build,
    'rope': rope_build,
    'cumsum': cumsum_build,
    'moe_tkg': moe_tkg_build,
}
