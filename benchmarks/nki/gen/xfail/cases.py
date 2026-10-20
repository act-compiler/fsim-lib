# The xfail cases of the nki suite, as pure data.
# Each case is generated into benchmarks/nki/<name>/ by gen.py.
# A case breaks one trn2 memory limit at an explicit address;
# fsim-nki must reject it with its reject substring.


def case(kernel, reject):
    return {'name': f'xfail/{kernel}', 'kernel': kernel, 'seed': 0, 'reject': reject}


CASES = [
    case('sbuf_capacity', 'past the 229376 bytes per partition of sbuf'),
    case('partitions', 'past the 128 partitions of sbuf'),
    case('psum_capacity', 'past the 16384 bytes per partition of psum'),
    case('psum_bank', 'across a 2048-byte PSUM bank boundary'),
]
