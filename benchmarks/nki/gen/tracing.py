# Records a case's kernel into <case>/kernel.trace, the trace fsim-nki reads.
# Each gen calls record_trace once per case, right after its NEFF compile.
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# The recorder imports its own modules by name.
sys.path.append(os.path.join(REPO, 'drivers/nki/record'))
from record import record


def record_trace(b, folder):
    # Record the kernel's trace for fsim-nki; returns its seconds.
    t0 = time.perf_counter()
    record(b, os.path.join(folder, 'kernel.trace'))
    return time.perf_counter() - t0
