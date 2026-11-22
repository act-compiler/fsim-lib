# Generates the nkilib cases into benchmarks/nki/<name>/.
# Each case folder gets kernel.neff, inputs/<tensor>.bin and golden/<tensor>.bin,
# named after the kernel's tensors; compiler artifacts go to a per-case work dir, then discarded.
# Each .bin holds its tensor of each of the case's M seeded input sets, back to back;
# input set 0 alone is compiled and recorded.
# Run in the nki env with its bin on PATH; the first failing case stops it.
#   gen.py [names...]    every case when none is named
import glob
import math
import os
import shutil
import sys
import tempfile
import time

import neuron_dtypes as ndt
import numpy as np
import torch
from cases import CASES
from kernels import BUILDERS, Rand, single, to_torch
from nki.compiler.driver import compile_to_bir
from nki.compiler.frontend import TracerFrontend
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

# The shared trace step, in gen/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tracing import record_trace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Fixed per-case compile dir, so the NEFF's recorded path never varies.
WORK = os.path.join(tempfile.gettempdir(), 'nkilib-gen')
# Input-set count M by one input set's inputs plus golden bytes: (under bytes, M).
INPUT_SET_COUNTS = [(1 << 20, 32), (16 << 20, 16), (64 << 20, 8), (256 << 20, 4), (math.inf, 2)]


def tensor_file(case_dir, sub, name):
    # The one naming rule: <case>/<inputs|golden|out>/<tensor name>.bin.
    return os.path.join(case_dir, sub, f'{name}.bin')


def np_dtype(name):
    return np.dtype(getattr(ndt, name, name))


def compile_neff(b, tmp):
    opts = CompileOptions(
        target='trn2', lnc=1, artifacts_dir=f'{tmp}/artifacts', output_path=f'{tmp}/kernel.neff'
    )
    bir = compile_to_bir(
        b.kernel,
        frontend=TracerFrontend(),
        inputs={**b.tensors, **b.kwargs},
        compile_opts=opts,
        output_names=['out'],
        enable_cache=False,
    )
    k = compile_bir_to_neff(opts, bir, list(b.tensors.values()), list(b.tensors), ['out'])
    # The NEFF and the spec of its one output.
    (spec,) = k.output_specs
    return k.neff_path, spec


def golden(b, dtype, shape):
    res = single(b.ref({n: to_torch(a) for n, a in b.tensors.items()}))
    if isinstance(res, torch.Tensor):
        res = res.detach().float().numpy()
    res = np.asarray(res)
    # Shapes agree but for unit dimensions, which refs may drop.
    if [d for d in res.shape if d != 1] != [d for d in shape if d != 1]:
        raise ValueError(f'golden has {res.shape}, NEFF output has {shape}')
    # Values are cast to the NEFF dtype.
    return res.astype(dtype).reshape(shape)


def append(path, a):
    # Input set s's tensor lands at s times one tensor's bytes.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'ab') as f:
        np.ascontiguousarray(a).tofile(f)


def signature(b):
    # What input set 0's record fixes: kernel, kwargs, tensor names, dtypes and shapes.
    return b.kernel, b.kwargs, [(n, a.dtype, a.shape) for n, a in b.tensors.items()]


def generate(c):
    out = os.path.join(ROOT, c['name'])
    # A failed case leaves no stale files or run behind.
    for d in ['inputs', 'golden', 'out']:
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
    if os.path.exists(os.path.join(out, 'kernel.neff')):
        os.remove(os.path.join(out, 'kernel.neff'))
    for log in glob.glob(os.path.join(out, '*.log')):
        os.remove(log)
    b = BUILDERS[c['kernel']](c, Rand(c['seed']))
    work = os.path.join(WORK, c['name'])
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        t0 = time.perf_counter()
        neff, spec = compile_neff(b, work)
        t_compile = time.perf_counter() - t0
        os.makedirs(out, exist_ok=True)
        shutil.copyfile(neff, os.path.join(out, 'kernel.neff'))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    t_record = record_trace(b, out)
    t0 = time.perf_counter()
    dt, shape = np_dtype(spec.dtype), tuple(spec.shape)
    fixed = signature(b)
    # The input-set count M by one input set's inputs and golden bytes.
    nbytes = sum(a.nbytes for a in b.tensors.values()) + dt.itemsize * math.prod(shape)
    input_sets = next(n for bound, n in INPUT_SET_COUNTS if nbytes < bound)
    for s in range(input_sets):
        if s:
            # A later input set has only its tensor contents regenerated, from seed + s.
            b = BUILDERS[c['kernel']](c, Rand(c['seed'] + s))
            if signature(b) != fixed:
                raise ValueError(
                    f'input set {s} differs from input set 0 in kernel, kwargs, dtypes or shapes'
                )
        for n, a in b.tensors.items():
            append(tensor_file(out, 'inputs', n), a)
        append(tensor_file(out, 'golden', spec.name), golden(b, dt, shape))
        # One input set in memory at a time.
        b = a = None
    t_input_sets = time.perf_counter() - t0
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    tensors = {n: ('input', t, dims) for n, t, dims in fixed[2]}
    tensors[spec.name] = ('output', dt, shape)
    io_desc = ' '.join(f'{k[0]}:{n}:{dt}{list(s)}' for n, (k, dt, s) in sorted(tensors.items()))
    neff_b = os.path.getsize(os.path.join(out, 'kernel.neff'))
    trace_b = os.path.getsize(os.path.join(out, 'kernel.trace'))
    print(
        f'OK {c["name"]} compile_s={t_compile:.1f} neff_b={neff_b} '
        f'record_s={t_record:.1f} trace_b={trace_b} '
        f'input_sets={input_sets} input_sets_s={t_input_sets:.2f} disk_b={size} {io_desc}',
        flush=True,
    )


if __name__ == '__main__':
    names = sys.argv[1:]
    known = {c['name'] for c in CASES}
    unknown = [n for n in names if n not in known]
    if unknown:
        sys.exit(f'unknown cases: {unknown}')
    for c in CASES:
        if names and c['name'] not in names:
            continue
        # Any failure propagates, naming its case.
        try:
            generate(c)
        except Exception as e:
            e.add_note(f'case {c["name"]}')
            raise
