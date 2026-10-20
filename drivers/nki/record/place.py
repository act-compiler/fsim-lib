# Placement for tensors the kernel gives no address.
# Each arena places a tensor in its lowest aligned gap between live
# blocks, above the floor; a release frees the block.

# trn2's bytes per partition and PSUM bank size.
SBUF = 229376
PSUM = 16384
PSUM_BANK = 2048


class Arena:
    def __init__(self, name, capacity, floor=0):
        self.name = name
        self.capacity = capacity
        self.floor = floor
        # Live blocks, [start, end], by start.
        self.blocks = []

    def place(self, size, align):
        # The lowest aligned start that fits a gap.
        start, at = self.floor, len(self.blocks)
        for i, (lo, hi) in enumerate(self.blocks):
            if -(-start // align) * align + size <= lo:
                at = i
                break
            start = max(start, hi)
        start = -(-start // align) * align
        if start + size > self.capacity:
            raise MemoryError(
                f'{self.name}: {size} B at {start} passes its {self.capacity} B capacity'
            )
        block = [start, start + size]
        self.blocks.insert(at, block)
        return block

    def release(self, block):
        self.blocks = [b for b in self.blocks if b is not block]


def arenas(hbm, floors):
    # The three arenas at trn2's sizes, hbm at the header's.
    sizes = {'hbm': hbm, 'sbuf': SBUF, 'psum': PSUM}
    return {a: Arena(a, n, floors.get(a, 0)) for a, n in sizes.items()}


def align(arena, size):
    # The dtype's alignment; in PSUM a bank, so no bank straddles.
    return PSUM_BANK if arena == 'psum' else size
