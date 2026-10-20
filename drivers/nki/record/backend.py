# The recorder backend: nki's builtins as trace records, nothing computed.
# _activate_backend installs a Recorder; an allocation is placed and
# written as an alloc record, a tensor builtin as one instruction record
# of its keyword fields. Control and dtype helpers are the simulator's.
import enum
import inspect
import math
import numbers
import weakref

import nki._backends.simulator as sim
import place
from nki.language.opcode import _Op

# Buffers to arenas.
ARENA = {
    'hbm': 'hbm',
    'shared_hbm': 'hbm',
    'private_hbm': 'hbm',
    'sbuf': 'sbuf',
    'psum': 'psum',
}
# nki dtypes to record dtypes, and their sizes.
DTYPE = {
    'bfloat16': 'bf16',
    'float16': 'f16',
    'float32': 'f32',
    'int32': 'i32',
    'uint8': 'u8',
    'uint16': 'u16',
    'uint32': 'u32',
}
SIZE = {'bf16': 2, 'f16': 2, 'f32': 4, 'i32': 4, 'u8': 1, 'u16': 2, 'u32': 4}
# Fields without semantics.
DROP = {'name', 'priority', 'dge_mode'}

# Simulator builtins kept as they are: control, targets, dtypes.
PLUMBING = {
    'get_current_context',
    'get_nc_version',
    'get_nc_sub_version',
    'get_lower_dma_transpose',
    'identity_matrix_cache',
    'enter_insert_at_beginning',
    'exit_insert_at_beginning',
    'insert_at_beginning_context',
    'no_reorder',
    'add_schedule_dependency',
    'num_programs',
    'program_id',
    'program_ndim',
    'affine_range',
    'sequential_range',
    'static_range',
    'dynamic_range',
    'fori_loop',
    'pipeline_loop',
    'while_loop',
    'register_alloc',
    'from_numpy_dtype',
    'to_numpy_dtype',
}
# Builtins the format cannot carry.
REFUSED = {
    'load_register',
    'store_register',
    'alloc_constant',
    'shared_constant',
    'inline_asm_bytes',
    'device_print',
    'sendrecv',
    'core_barrier',
    'rank_id',
    'all_reduce',
    'all_gather',
    'reduce_scatter',
    'all_to_all',
    'all_to_all_v',
    'all_gather_v',
    'collective_permute',
    'collective_permute_implicit',
    'collective_permute_implicit_reduce',
    'collective_permute_implicit_current_processing_rank_id',
}


class Tensor:
    """One allocation: its arena, address, dtype and shape."""

    def __init__(self, id, arena, partition, byte, dtype, shape, name):
        self.id = id
        self.tensor_id = id
        self.arena = arena
        self.partition = partition
        self.byte = byte
        self.dtype = dtype
        self.shape = tuple(shape)
        self.name = name
        # Frees a driver placement, on release or collection.
        self.free = None
        # The other slots a multi-buffer rotates through.
        self.slots = []
        self.turn = 0

    def bytes(self):
        # Bytes per partition, or in all for hbm.
        n = math.prod(self.shape if self.arena == 'hbm' else self.shape[1:])
        return n * SIZE[self.dtype]


class View:
    """An operand as its record text."""

    def __init__(self, text):
        self.text = text


class Affine:
    """An affine map: a pattern and its offset."""

    def __init__(self, pattern, offset):
        self.pattern = pattern
        self.offset = offset


def dtype(d):
    name = str(d)
    if name not in DTYPE:
        raise NotImplementedError(f'dtype {name} is not recorded')
    return DTYPE[name]


def pattern_text(p):
    return '[' + ','.join(f'[{int(s)},{int(n)}]' for s, n in p) + ']'


def word(s):
    # A token: no spaces, no field separators.
    if not s or any(c.isspace() or c == '=' for c in s):
        raise ValueError(f'{s!r} is not a record word')
    return s


class Recorder:
    """The backend: writes each record through write, a line callable.

    Driver placements start at floors, per arena, above kernel addresses.
    """

    def __init__(self, write, hbm, floors):
        self.write = write
        self.arenas = place.arenas(hbm, floors)
        self.ids = 0
        # Each arena's end of its kernel-placed bytes.
        self.kernel_top = {}
        # Stack tensors per open scope.
        self.scopes = []

    def __getattr__(self, kind):
        # Builtins not defined here: plumbing, refused, or an instruction.
        builtin = getattr(sim, kind)
        if kind in PLUMBING:
            return builtin
        if kind.startswith('register_to_py'):
            raise NotImplementedError(f'{kind}: a register is no Python value at record time')
        if kind in REFUSED or not callable(builtin):
            raise NotImplementedError(f'{kind} is not recorded')
        signature = inspect.signature(builtin)

        def instruction(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            fields = [f'{k}={self.value(v)}' for k, v in bound.arguments.items() if k not in DROP]
            self.write(' '.join([kind, *fields]))

        return instruction

    def value(self, v):
        if isinstance(v, View):
            return v.text
        if isinstance(v, Affine):
            return f'aff({pattern_text(v.pattern)},{int(v.offset)})'
        if v is None:
            return 'none'
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, numbers.Integral):
            return str(int(v))
        if isinstance(v, numbers.Real):
            return repr(float(v))
        if isinstance(v, enum.Enum):
            return word(v.name)
        if isinstance(v, _Op):
            return word(v._name)
        if isinstance(v, (list, tuple)) and all(isinstance(x, numbers.Integral) for x in v):
            return '[' + ','.join(str(int(x)) for x in v) + ']'
        if isinstance(v, str):
            return word(v)
        raise NotImplementedError(f'value {v!r} of {type(v).__name__} is not recorded')

    # Allocations

    def alloc(self, shape, dt, buffer, name, address, input=None):
        # Places one tensor, then writes its record.
        arena = ARENA[getattr(buffer, 'name', buffer)]
        t = Tensor(self.ids, arena, 0, 0, dtype(dt), shape, name or None)
        self.ids += 1
        if address is None:
            # A tensor the kernel can no longer name is dead.
            a = self.arenas[arena]
            block = a.place(t.bytes(), place.align(arena, SIZE[t.dtype]))
            t.byte = block[0]
            t.free = weakref.finalize(t, a.release, block)
        else:
            if arena == 'hbm':
                t.byte = int(address)
            else:
                t.partition, t.byte = (int(a) for a in address)
            top = max(self.kernel_top.get(arena, 0), t.byte + t.bytes())
            self.kernel_top[arena] = top
        addr = str(t.byte) if arena == 'hbm' else f'{t.partition},{t.byte}'
        dims = ','.join(str(int(d)) for d in t.shape)
        line = f'alloc %{t.id} {arena}@{addr} {t.dtype}[{dims}]'
        line += ' kernel' if address is not None else ' driver'
        if t.name:
            line += f' name={word(t.name)}'
        if input:
            line += f' input={word(input)}'
        self.write(line)
        return t

    def alloc_tensor(
        self, shape, dtype, buffer, name, address, is_static=False, num_slots=1, auto_release=True
    ):
        if num_slots > 1 and address is not None:
            raise NotImplementedError(
                f'num_slots={num_slots} at an explicit address is not recorded'
            )
        t = self.alloc(shape, dtype, buffer, name, address)
        t.slots = [self.alloc(shape, dtype, buffer, name, address) for _ in range(num_slots - 1)]
        return t

    def alloc_stack_tensor(self, shape, dtype, buffer, name):
        # Released, in stack order, when its scope closes.
        t = self.alloc(shape, dtype, buffer, name, None)
        self.scopes[-1].append(t)
        return t

    def rotate_tensor(self, tensor):
        slots = [tensor, *tensor.slots]
        slot = slots[tensor.turn % len(slots)]
        tensor.turn += 1
        return slot

    def release_tensor(self, tensor):
        for t in [tensor, *tensor.slots]:
            if t.free:
                t.free()

    def open_scope(self):
        self.scopes.append([])

    def close_scope(self):
        for t in reversed(self.scopes.pop()):
            self.release_tensor(t)

    # Operands

    def tensor_ap_view(
        self,
        tensor,
        tensor_shape,
        tensor_dtype,
        pattern,
        offset=0,
        scalar_offset=None,
        vector_offset=None,
        indirect_dim=0,
        ti_index=None,
        ti_num_elem=None,
    ):
        # The view over its allocation's row, as arena@addr:dtype[pattern]dyn.
        if vector_offset is not None or ti_index is not None:
            raise NotImplementedError('vector offsets are not recorded')
        dt = dtype(tensor_dtype)
        size = SIZE[dt]
        offset = int(offset)
        dyn = ''
        if scalar_offset is not None:
            # A scalar offset steps the indirect dimension.
            k = math.prod(tensor_shape[indirect_dim + 1 :])
            if isinstance(scalar_offset, View):
                if tensor.arena != 'hbm' and indirect_dim == 0:
                    raise NotImplementedError('a +s() offset across partitions is defined at M3')
                dyn = f'+s({scalar_offset.text})*{k}'
            else:
                offset += k * int(scalar_offset)
        pat = [[int(s), int(n)] for s, n in pattern]
        if tensor.arena == 'hbm':
            addr = str(tensor.byte + offset * size)
        else:
            # Partition stride and offset in rows of the allocation.
            row, rem = divmod(tensor.bytes(), size)
            if rem or pat[0][0] % row:
                raise ValueError(
                    f'%{tensor.id}: partition stride {pat[0][0]} is not whole rows of {row} {dt}'
                )
            p, e = divmod(offset, row)
            addr = f'{tensor.partition + p},{tensor.byte + e * size}'
            pat[0][0] //= row
        return View(f'{tensor.arena}@{addr}:{dt}{pattern_text(pat)}{dyn}')

    def build_affine_value_map(self, pattern, static_offset=0, dynamic_offset=None):
        # A register offset must be a known int.
        return Affine(pattern, static_offset + int(dynamic_offset or 0))
