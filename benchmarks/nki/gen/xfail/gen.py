# Generates the xfail cases into benchmarks/nki/<name>/.
# Each case folder gets inputs/<tensor>.bin and no golden, since fsim-nki must reject it.
# Each kernel is also compiled for trn2 and run under nki.simulate, and both are logged;
# a kernel the compiler accepts fails its case.
# Run in the nki env with its bin on PATH; the first failing case stops it.
#   gen.py [names...]    every case when none is named
import contextlib
import glob
import os
import re
import shutil
import sys
import tempfile
import time

# nki.simulate's target, read as nki loads.
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import numpy as np
from cases import CASES
from kernels import BUILDERS, Rand
from nki.compiler.driver import compile_to_bir
from nki.compiler.frontend import TracerFrontend
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

# The shared trace step, in gen/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tracing import record_trace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WORK = os.path.join(tempfile.gettempdir(), 'xfail-gen')


def tensor_file(case_dir, sub, name):
    # The one naming rule: <case>/<inputs|golden|out>/<tensor name>.bin.
    return os.path.join(case_dir, sub, f'{name}.bin')


def write(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.ascontiguousarray(a).tofile(path)


@contextlib.contextmanager
def stderr_to(path):
    # Native stderr, where MLIR diagnostics go, into a file.
    sys.stderr.flush()
    saved = os.dup(2)
    with open(path, 'w') as f:
        os.dup2(f.fileno(), 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


def reason(text):
    # The first message, without error codes or internal names.
    m = re.search(r'\[NCC_\w+\] (.*?)(?: - Please|$)', text, re.MULTILINE)
    m = m or re.search(r'error: (.*)$', text, re.MULTILINE)
    msg = m.group(1) if m else text.strip().splitlines()[0]
    return msg.removeprefix('Unhandled exception with message: ').split(' {')[0]


def compile_reject(b, tmp):
    # The compiler's rejection, as gen/nkilib/gen.py compiles; None if accepted.
    opts = CompileOptions(
        target='trn2', lnc=1, artifacts_dir=f'{tmp}/artifacts', output_path=f'{tmp}/kernel.neff'
    )
    log = os.path.join(tmp, 'stderr.log')
    try:
        with stderr_to(log):
            bir = compile_to_bir(
                b.kernel,
                frontend=TracerFrontend(),
                inputs={**b.tensors, **b.kwargs},
                compile_opts=opts,
                output_names=['out'],
                enable_cache=False,
            )
            compile_bir_to_neff(opts, bir, list(b.tensors.values()), list(b.tensors), ['out'])
    except Exception as e:
        with open(log) as f:
            return reason(f'{e}\n{f.read()}')
    return None


def simulate_reject(b):
    # nki.simulate's rejection; None if the kernel runs.
    try:
        nki.simulate(b.kernel)(**{n: a.copy() for n, a in b.tensors.items()}, **b.kwargs)
    except Exception as e:
        return reason(str(e))
    return None


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
        compiled = compile_reject(b, work)
        t_compile = time.perf_counter() - t0
    finally:
        shutil.rmtree(work, ignore_errors=True)
    t_record = record_trace(b, out)
    if compiled is None:
        raise ValueError('the compiler accepts the kernel, so it breaks no limit')
    for n, a in b.tensors.items():
        write(tensor_file(out, 'inputs', n), a)
    simulated = simulate_reject(b)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    io_desc = ' '.join(f'i:{n}:{a.dtype}{list(a.shape)}' for n, a in sorted(b.tensors.items()))
    sim = 'simulate=runs' if simulated is None else f'simulate_reject="{simulated}"'
    trace_b = os.path.getsize(os.path.join(out, 'kernel.trace'))
    print(
        f'OK {c["name"]} compile_s={t_compile:.1f} record_s={t_record:.1f} trace_b={trace_b} '
        f'disk_b={size} {io_desc} '
        f'{sim} compile_reject="{compiled}"',
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
