# Generates the looped matmul cases into benchmarks/nki/<name>/.
# Each case gets kernel.neff and kernel.trace from kernels.py, seeded inputs,
# and a golden accumulated in fp32, all named by the kernel's tensors.
#   gen.py [names...]    every case when none is named
import glob
import os
import shutil
import sys
import tempfile
import time

import torch
from cases import CASES
from kernels import Build, P, matmul
from nki.compiler.driver import compile_to_bir
from nki.compiler.frontend import TracerFrontend
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

# The shared trace step, in gen/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tracing import record_trace

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(os.path.dirname(HERE))
# Fixed per-case compile dir, so the NEFF's recorded path never varies.
WORK = os.path.join(tempfile.gettempdir(), 'matmul-gen')
# The kernel's one output.
OUTPUT = 'tensor_c'


def write(path, t):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.contiguous().numpy().tofile(path)


def compile_neff(b, name, out):
    # Compiles the kernel for trn2 into out/kernel.neff.
    work = os.path.join(WORK, name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        opts = CompileOptions(
            target='trn2',
            lnc=1,
            artifacts_dir=f'{work}/artifacts',
            output_path=f'{work}/kernel.neff',
        )
        bir = compile_to_bir(
            b.kernel,
            frontend=TracerFrontend(),
            inputs={**b.tensors, **b.kwargs},
            compile_opts=opts,
            output_names=[OUTPUT],
            enable_cache=False,
        )
        k = compile_bir_to_neff(opts, bir, list(b.tensors.values()), list(b.tensors), [OUTPUT])
        shutil.copyfile(k.neff_path, os.path.join(out, 'kernel.neff'))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def name_output(folder):
    # The recorder names its one output out; the trace names it OUTPUT.
    path = os.path.join(folder, 'kernel.trace')
    with open(path) as f:
        text = f.read()
    if text.count('\noutput out %') != 1:
        raise ValueError(f'{path}: not one output out record')
    with open(path, 'w') as f:
        f.write(text.replace('\noutput out %', f'\noutput {OUTPUT} %'))


def generate(case):
    # One case: NEFF, trace, inputs, golden.
    m, k, n = case['M'], case['K'], case['N']
    dtype = getattr(torch, case['dtype'])
    out = os.path.join(SUITE, case['name'])
    # The last run predates this NEFF.
    shutil.rmtree(os.path.join(out, 'out'), ignore_errors=True)
    for log in glob.glob(os.path.join(out, '*.log')):
        os.remove(log)
    os.makedirs(out, exist_ok=True)
    # Uniform [0, 1) inputs, A then B from one seed.
    g = torch.Generator().manual_seed(case['seed'])
    a = torch.rand(m, k, generator=g).to(dtype)
    b = torch.rand(k, n, generator=g).to(dtype)
    ident = torch.eye(P, dtype=dtype)
    build = Build(
        matmul, {'tensor_a': a.numpy(), 'tensor_b': b.numpy(), 'tensor_id': ident.numpy()}, {}
    )
    compile_neff(build, case['name'], out)
    record_trace(build, out)
    name_output(out)
    write(os.path.join(out, 'inputs', 'tensor_a.bin'), a)
    write(os.path.join(out, 'inputs', 'tensor_b.bin'), b)
    write(os.path.join(out, 'inputs', 'tensor_id.bin'), ident)
    # Golden: fp32 accumulate, cast to the output dtype.
    c = (a.float() @ b.float()).to(dtype)
    write(os.path.join(out, 'golden', 'tensor_c.bin'), c)
    return out


def du(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


if __name__ == '__main__':
    by_name = {c['name']: c for c in CASES}
    unknown = [n for n in sys.argv[1:] if n not in by_name]
    if unknown:
        sys.exit('unknown cases: ' + ' '.join(unknown))
    todo = [by_name[n] for n in sys.argv[1:]] or CASES
    # Any failure raises and exits nonzero.
    for case in todo:
        t0 = time.perf_counter()
        out = generate(case)
        dt = time.perf_counter() - t0
        print(f'{os.path.relpath(out)}: {dt:.1f} s, {du(out) / 1e6:.1f} MB', flush=True)
