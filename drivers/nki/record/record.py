# Records one kernel into a trace: the header, the records the
# backend writes as the kernel runs, its outputs and the end.
import math

import backend
import nki._backends as backends
from nki._backends.simulator import from_numpy_dtype
from nki._backends.simulator.state import SimulatorContext, current_context
from nki.language.buffers import shared_hbm
from nki.language.tensor import NkiTensor

HEADER = 'nki-trace 1 target=trn2 lnc=1 hbm={}'


def run(build, write, hbm, floors):
    # One pass: places the inputs, then runs the kernel.
    rec = backend.Recorder(write, hbm, floors)
    kernel = getattr(build.kernel, 'func', build.kernel)
    ctx = SimulatorContext(target='trn2', precise_fp=True)
    with backends._activate_backend(rec), current_context(ctx):
        args = {}
        for name, a in build.tensors.items():
            dt = from_numpy_dtype(a.dtype)
            t = rec.alloc(a.shape, dt, shared_hbm, None, None, input=name)
            args[name] = NkiTensor(shape=a.shape, dtype=dt, buffer=shared_hbm, storage=t)
        result = kernel(**args, **build.kwargs)
    results = result if isinstance(result, (tuple, list)) else [result]
    if len(results) != 1:
        raise NotImplementedError(f'{len(results)} outputs; one, out, is recorded')
    return results[0]._storage, rec.kernel_top


def record(build, out, reserve=None):
    """Records build's kernel into out, a path or a text stream.

    HBM holds the kernel's placements, the inputs, the outputs and reserve
    bytes of scratch, by default the inputs' and outputs' size.
    Returns [(name, tensor)] for its one output, out.
    """
    if isinstance(out, str):
        with open(out, 'w') as f:
            return record(build, f, reserve)
    # A dry pass sizes the output and kernel placements.
    inputs = sum(a.nbytes for a in build.tensors.values())
    t, floors = run(build, lambda line: None, math.inf, {})
    io = inputs + t.bytes()
    # Driver placements start above the kernel's highest HBM byte.
    hbm = floors.get('hbm', 0) + io + (io if reserve is None else reserve)
    # The f32 grid wants whole 4-byte words.
    hbm = -(-hbm // 4) * 4

    def write(line):
        out.write(line + '\n')

    write(HEADER.format(hbm))
    t, _ = run(build, write, hbm, floors)
    write(f'output out %{t.id}')
    write('end')
    out.flush()
    return [('out', t)]
