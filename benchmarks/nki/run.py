# Runs one fsim-nki build over each case and checks every output.
# A run writes <case>/out/ and <case>/<log>, named by its build and knobs,
# whose last line is its status: match, wrong, failing, N/A or not generated.
#   run.py [--compiler C] [--variant V] [--trigger N] [--overlap T] [bench...]
# FSIM_WALL_TIMEOUT caps each run's wall seconds, 60 by default.
import argparse
import ctypes
import os
import re
import shutil
import signal
import subprocess
import sys
import traceback

import ml_dtypes
import numpy as np
import suite

REPO = os.path.dirname(os.path.dirname(suite.ROOT))
BUILD = os.path.join(REPO, 'drivers/nki/build')
# Builds: compilers by variants, and each variant's knobs.
COMPILERS = [
    'xla-0.11-cpu',
    'xla-0.4-cpu',
    'custom-cpu',
    'xla-0.11-gpu',
    'xla-0.4-gpu',
    'custom-gpu',
]
VARIANTS = ['eager', 'taidl', 'trigger', 'overlap']
KNOBS = {'eager': [], 'taidl': [], 'trigger': ['trigger'], 'overlap': ['trigger', 'overlap']}
DEFAULTS = {'trigger': 10000, 'overlap': 4}
# Wall seconds per run, the development cap.
TIMEOUT = float(os.environ.get('FSIM_WALL_TIMEOUT', '60'))

# Log lines failing a run on any exit code; first match wins.
FAILURES = [
    (re.compile(p, re.MULTILINE), cause)
    for p, cause in [
        (r'^fsim-nki: (.+)$', r'fsim-nki: \1'),
        (r'^fsim: (.+)$', r'fsim: \1'),
    ]
]

# Per dtype: numpy type, max relative error, floor as a fraction of max |ref|.
TOLERANCE = {
    'float32': (np.float32, 1e-3, 1e-2),
    'float16': (np.float16, 1e-2, 1e-1),
    'bfloat16': (ml_dtypes.bfloat16, 3e-2, 1e-1),
}

# A taidl build's stop at a mid-run flush, which makes the case N/A.
MIDRUN = re.compile(r'fsim: a second module')

# fsim's summary line of a run, the last one in its log.
SUMMARY = re.compile(r'^fsim stats: (.*)$', re.MULTILINE)
# A run's times in seconds, in the order the summary shows them.
TIMES = ['tracing', 'overhead', 'simulation', 'tracing_overhead', 'compile_overhead', 'wall']
# A run's counts, after its times.
COUNTS = ['trace', 'modules']


def sim(compiler, variant):
    return os.path.join(BUILD, f'{compiler}-{variant}', 'fsim-nki')


def gpus():
    # The CUDA driver's visible device count, 0 without a driver.
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
    except OSError:
        return 0
    count = ctypes.c_int()
    if cuda.cuInit(0) or cuda.cuDeviceGetCount(ctypes.byref(count)):
        return 0
    return count.value


def log_name(compiler, variant, knobs):
    # <compiler>-<variant>[-<trigger>[-<overlap>]].log, a knob per one it takes.
    names = [compiler, variant] + [str(knobs[k]) for k in KNOBS[variant]]
    return '-'.join(names) + '.log'


def bins(folder, sub):
    d = os.path.join(folder, sub)
    return sorted(f for f in os.listdir(d) if f.endswith('.bin')) if os.path.isdir(d) else []


def missing(folder, case):
    # What generation left out, or None.
    if not os.path.isfile(os.path.join(folder, 'kernel.trace')):
        return 'no kernel.trace'
    if not bins(folder, 'inputs'):
        return 'no inputs'
    if not bins(folder, 'golden'):
        return 'no golden'
    return None


def simulate(folder, log, timeout, compiler, variant, knobs):
    # One simulator run into out/; returns its failure cause, or None.
    out = os.path.join(folder, 'out')
    if os.path.isdir(out):
        shutil.rmtree(out)
    cmd = [sim(compiler, variant), '--inputs', os.path.join(folder, 'inputs'), '--outputs', out]
    for k in KNOBS[variant]:
        cmd += [f'--{k}', str(knobs[k])]
    cmd += [os.path.join(folder, 'kernel.trace')]
    limits = f'FSIM_WALL_TIMEOUT={timeout:g}'
    log.write(f'{limits} {" ".join(cmd)}\n')
    log.flush()
    # Its own process group, so its children die with it.
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        kill_group(proc)
    with open(log.name, errors='replace') as f:
        text = f.read()
    # The taidl stop comes first, whatever else it logs.
    if variant == 'taidl' and MIDRUN.search(text):
        return 'mid-run flush'
    for pattern, cause in FAILURES:
        m = pattern.search(text)
        if m:
            return m.expand(cause)
    if timed_out:
        return 'timeout'
    if proc.returncode < 0:
        return f'signal {signal.Signals(-proc.returncode).name}'
    if proc.returncode:
        return f'exit {proc.returncode}'
    return None


def kill_group(proc):
    # SIGKILLs every process left in proc's group, then reaps them.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # the group already emptied
    proc.wait()
    # As pid 1 in the container, the group's orphans are ours to reap.
    try:
        while True:
            os.waitpid(-proc.pid, 0)
    except ChildProcessError:
        pass  # none of the group left


def rel_error(ref, act, floor):
    # Per element |r - a| / max(|r|, |a|, floor * max |r|), every element checked.
    finite = np.abs(ref[np.isfinite(ref)])
    diff = np.where(ref == act, 0.0, np.abs(ref - act))
    denom = np.maximum(np.maximum(np.abs(ref), np.abs(act)), floor * finite.max(initial=0.0))
    return np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)


def compare(folder, case):
    # Every golden against its output; returns (cause or None, max error).
    worst, over = 0.0, False
    kind, tol, floor = TOLERANCE[case['dtype']]
    for name in bins(folder, 'golden'):
        golden = os.path.join(folder, 'golden', name)
        out = os.path.join(folder, 'out', name)
        if not os.path.isfile(out):
            return f'no out/{name}', None
        size, want = os.path.getsize(out), os.path.getsize(golden)
        if size != want:
            return f'out/{name} is {size} B, golden {want} B', None
        ref = np.fromfile(golden, kind).astype(np.float64)
        act = np.fromfile(out, kind).astype(np.float64)
        if ((~np.isfinite(ref) | ~np.isfinite(act)) & (ref != act)).any():
            return f'out/{name} has nan or inf', None
        if ref.any() and not act.any():
            return f'out/{name} is all zero', None
        rel = rel_error(ref, act, floor)
        if np.isnan(rel).any():
            return f'out/{name} has a nan error', None
        worst = max(worst, float(rel.max()))
        over |= rel.max() > tol
    return ('over tolerance' if over else None), worst


def ends_line(path):
    # Whether the file is empty or ends with a newline.
    with open(path, 'rb') as f:
        if f.seek(0, os.SEEK_END) == 0:
            return True
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b'\n'


def times(path):
    # TIMES then COUNTS from the last summary; '-' without one.
    with open(path, errors='replace') as f:
        lines = SUMMARY.findall(f.read())
    found = dict(field.split('=', 1) for field in lines[-1].split()) if lines else {}
    return {k: found.get(k, '-') for k in TIMES + COUNTS}


def run(case, timeout, compiler, variant, knobs):
    # One case end to end; returns its status line.
    folder = os.path.join(suite.ROOT, case['name'])
    path = os.path.join(folder, log_name(compiler, variant, knobs))
    err, spent = '-', dict.fromkeys(TIMES + COUNTS, '-')
    with open(path, 'w') as log:
        status = None
        try:
            cause = missing(folder, case)
            if cause:
                status = 'not generated'
            else:
                cause = simulate(folder, log, timeout, compiler, variant, knobs)
                spent = times(path)
                if cause:
                    status = 'N/A' if cause == 'mid-run flush' else 'failing'
                else:
                    cause, worst = compare(folder, case)
                    status = 'wrong' if cause else 'match'
                    err = '-' if worst is None else f'{worst:.3e}'
        except Exception as e:
            # A runner error fails the case, then propagates.
            traceback.print_exc(file=log)
            status, cause = 'failing', f'run.py: {type(e).__name__}: {e}'.replace('\n', ' ')
            raise
        finally:
            # One status line, none only when interrupted.
            if status:
                line = f'status: {status}' + (f' ({cause})' if cause else '')
                line += f' err={err} ' + ' '.join(f'{k}={v}' for k, v in spent.items())
                # A killed run's last line may be cut short.
                log.flush()
                if not ends_line(path):
                    log.write('\n')
                log.write(line + '\n')
    return status, line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nki cases.')
    parser.add_argument('benches', nargs='*', help='case name, directory or name prefix')
    parser.add_argument(
        '--compiler', choices=COMPILERS, default='custom-cpu', help='build compiler'
    )
    parser.add_argument('--variant', choices=VARIANTS, default='overlap', help='build variant')
    parser.add_argument(
        '--trigger', type=int, help=f'flush op count, {DEFAULTS["trigger"]} by default'
    )
    parser.add_argument(
        '--overlap', type=int, help=f'compile threads, {DEFAULTS["overlap"]} by default'
    )
    args = parser.parse_args()
    todo = suite.resolve(suite.load(), args.benches)
    unknown = sorted({c['dtype'] for c in todo} - TOLERANCE.keys())
    if unknown:
        sys.exit('no tolerance for dtype: ' + ' '.join(unknown))
    # Knobs the variant takes; others refused, as the simulator does.
    given = {k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None}
    if 'trigger' in given and 'trigger' not in KNOBS[args.variant]:
        sys.exit(f'--trigger: the {args.variant} build has no soft trigger')
    if 'overlap' in given and 'overlap' not in KNOBS[args.variant]:
        sys.exit(f'--overlap: the {args.variant} build compiles on no threads')
    knobs = {**DEFAULTS, **given}
    binary = sim(args.compiler, args.variant)
    if not os.path.isfile(binary):
        build = f'./scripts/build.sh nki --compiler {args.compiler} --variant {args.variant}'
        sys.exit(f'no simulator {os.path.relpath(binary, REPO)}: run {build}')
    # A gpu build needs a visible GPU before any case runs.
    if args.compiler.endswith('-gpu') and not gpus():
        sys.exit(f'no visible GPU: the {args.compiler} build runs on one')
    width = max(len(c['name']) for c in todo)
    statuses = []
    for case in todo:
        status, line = run(case, TIMEOUT, args.compiler, args.variant, knobs)
        statuses.append(status)
        print(f'{case["name"]:<{width}}  {line.removeprefix("status: ")}', flush=True)
    sys.exit(0 if all(s == 'match' for s in statuses) else 1)
