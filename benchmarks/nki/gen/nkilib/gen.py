# Generates the nkilib cases into benchmarks/nki/<name>/.
# Each case folder gets kernel.neff, inputs/<tensor>.bin and golden/<tensor>.bin,
# named after the kernel's tensors; compiler artifacts go to a per-case work dir, then discarded.
# Run in the nki env with its bin on PATH; the first failing case stops it.
#   gen.py [names...]    every case when none is named
import glob
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Fixed per-case compile dir, so the NEFF's recorded path never varies.
WORK = os.path.join(tempfile.gettempdir(), 'nkilib-gen')


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


def write(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.ascontiguousarray(a).tofile(path)


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
    for n, a in b.tensors.items():
        write(tensor_file(out, 'inputs', n), a)
    t0 = time.perf_counter()
    dt, shape = np_dtype(spec.dtype), tuple(spec.shape)
    write(tensor_file(out, 'golden', spec.name), golden(b, dt, shape))
    t_golden = time.perf_counter() - t0
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    tensors = {n: ('input', a.dtype, a.shape) for n, a in b.tensors.items()}
    tensors[spec.name] = ('output', dt, shape)
    io_desc = ' '.join(f'{k[0]}:{n}:{dt}{list(s)}' for n, (k, dt, s) in sorted(tensors.items()))
    neff_b = os.path.getsize(os.path.join(out, 'kernel.neff'))
    print(
        f'OK {c["name"]} compile_s={t_compile:.1f} neff_b={neff_b} '
        f'golden_s={t_golden:.2f} disk_b={size} {io_desc}',
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
