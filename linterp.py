# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

from contextlib import contextmanager
from typing import Any, Generator, cast
from dataclasses import dataclass, field

import ltypes
import opcodes

DEBUG = False

class Stack:

    def __init__(self):
        self.stacks = [[]]

    def push_bound(self):
        self.stacks.append([])

    def pop_bound(self):
        self.stacks.pop()

    def discard_bound(self):
        tmp = self.stacks.pop()
        self.stacks[-1] += tmp

    @contextmanager
    def push_stack_guard(self, top=0):
        self.push_bound()
        try:
            yield
        finally:
            if top == 0:
                self.pop_bound()
            elif top is None:
                self.discard_bound()
            else:
                self.set_top(top)
                self.discard_bound()

    def push(self, value):
        if not isinstance(value, ltypes.LuaType):
            raise ValueError(f"can only push lua values, got {value}")
        self.stacks[-1].append(value)

    def push_many(self, values):
        for value in values:
            self.push(value)

    def pop(self, expect=None, allow_ref=False):
        # TODO: Reimplement source info (local/upvalue/global)
        if not self.stacks[-1]:
            raise ltypes.InterpreterError("pop off empty substack")
        result = self.stacks[-1].pop()
        if isinstance(result, ltypes.Ref) and not allow_ref:
            raise ltypes.InterpreterError("pop reference")
        if expect is not None and result.type.s.decode("ascii") not in expect:
            raise ltypes.InterpreterError(f"expected {', '.join(expect)}, got {result.type.s.decode('ascii')}")
        return result

    def peek(self, idx, cross_boundary=False):
        if cross_boundary:
            offs = 0
            for stack in reversed(self.stacks):
                if idx in range(offs, offs + len(stack)):
                    return stack[-(idx-offs)-1]
                offs += len(stack)
            raise IndexError("index out of range")
        return self.stacks[-1][-idx-1]

    def set_top(self, arg):
        if arg < 0:
            del self.stacks[-1][arg:]
        elif arg > len(self.stacks[-1]):
            self.stacks[-1] += [ltypes.nil] * (arg - len(self.stacks[-1]))
        elif arg < len(self.stacks[-1]):
            del self.stacks[-1][arg-len(self.stacks[-1]):]

    def get_top(self):
        return len(self.stacks[-1])

    def __repr__(self):
        return "; ".join(" ".join(repr(j) for j in i) for i in self.stacks)

@dataclass
class Block:
    break_addr: int
    stack_level: int  # TODO: Redundant for everything but `for`
    locals: dict[bytes, ltypes.LuaType]

@dataclass
class ForeignFrame:
    thread: Any
    return_limit: int | None
    generator: Generator[None, None, tuple[ltypes.LuaType, ...] | ltypes.LuaType]

    stack: Stack = field(default_factory=Stack)
    to_throw: Exception | None = None

    def tick(self):
        try:
            if self.to_throw:
                to_throw = self.to_throw
                self.to_throw = None
                return self.generator.throw(to_throw)
            return next(self.generator)
        except StopIteration as e:
            err = e
        lim = self.return_limit
        self.thread.call_stack.pop()
        value = (*err.value,) if not isinstance(err.value, ltypes.LuaType) else (err.value,)
        if lim is None:
            for i in cast(tuple, value):
                self.thread.call_stack[-1].stack.push(i)
        else:
            for i, _ in zip(value, range(lim)):
                self.thread.call_stack[-1].stack.push(i)

@dataclass
class Frame:
    thread: Any

    strings: list[bytes]
    ops: list[tuple[opcodes.Opcode, int]]
    ip: int

    upvalues: dict[bytes, ltypes.Table | dict]  # each entry refers to a scope that contains the upvalue in question
    varargs: list[ltypes.LuaType]

    return_limit: int | None
    pcall_frame: bool

    block_stack: list[Block]

    stack: Stack = field(default_factory=Stack)

    def tick(self):
        opcode, offs = self.ops[self.ip]
        if DEBUG:
            call_stack_debug = "; ".join(f"[{','.join(j.decode("ascii", errors="backslashreplace") for i in frame.block_stack for j in i.locals)}],[{','.join(i.decode("ascii", errors="backslashreplace") for i in frame.upvalues.keys())}]" for frame in self.thread.call_stack if hasattr(frame, "block_stack"))
            print(f"{self.ip:04x}{' '*len(self.thread.call_stack)+str(opcode):<50} | {self.stack!r:<100} | {call_stack_debug}")

        match opcode:
            case opcodes.PushString(arg):
                self.stack.push(ltypes.String(arg))
            case opcodes.PushNumber(arg):
                self.stack.push(ltypes.Number(float(arg)))
            case opcodes.PushNil():
                self.stack.push(ltypes.nil)
            case opcodes.PushTrue():
                self.stack.push(ltypes.true)
            case opcodes.PushFalse():
                self.stack.push(ltypes.false)
            case opcodes.PushInline(arg):
                self.stack.push(ltypes.Code(self.ip + 1))
                self.ip += arg
            case opcodes.PushFunction(arg):
                # TODO: This may overcapture, test
                argnames = reversed([self.stack.pop(expect={"string"}).s for _ in range(arg)])
                offs = self.stack.pop().offs
                upvalues = {}
                for frame in self.thread.call_stack:
                    if isinstance(frame, ForeignFrame):
                        continue
                    upvalues.update(frame.upvalues)
                    for block in frame.block_stack:
                        for local in block.locals:
                            upvalues[local] = block.locals
                self.stack.push(ltypes.Function(self, tuple(argnames), offs, (self.strings, self.ops), upvalues))
            case opcodes.PushTable(arg):
                trailing_values = []
                while self.stack.get_top() > 0:
                    trailing_values.insert(0, self.stack.pop())
                self.stack.pop_bound()
                # TODO: Is this behavior 100% accurate
                # TODO: This doesn't differentiate between invalid nil keys and valid positional table constructor parameters
                arr = []
                for _ in range(arg):
                    v = self.stack.pop()
                    arr.append((self.stack.pop(), v))
                tab = ltypes.Table()
                for k, v in reversed(arr):
                    if k is ltypes.nil:
                        tab.seq.append(v)
                    else:
                        tab[k] = v
                tab.seq += trailing_values
                self.stack.push(tab)
            case opcodes.PushVarargs(arg):
                self.stack.push_many(self.varargs)
            case opcodes.RefName(arg):
                for block in self.block_stack[::-1]:
                    if arg in block.locals:
                        self.stack.push(ltypes.Ref(block.locals, arg))
                        break
                else:
                    if arg in self.upvalues:
                        self.stack.push(ltypes.Ref(self.upvalues[arg], arg))
                    else:
                        self.stack.push(ltypes.Ref(self.upvalues[b"_ENV"][b"_ENV"], ltypes.String(arg)))
            case opcodes.GetName(arg):
                for block in self.block_stack[::-1]:
                    if arg in block.locals:
                        self.stack.push(block.locals[arg])
                        break
                else:
                    if arg in self.upvalues:
                        self.stack.push(self.upvalues[arg][arg])
                    else:
                        self.stack.push(self.upvalues[b"_ENV"][b"_ENV"][ltypes.String(arg)])
            case opcodes.RefItem():
                self.stack.push(ltypes.Ref(*[self.stack.pop(), self.stack.pop(expect={"table"})][::-1]))
            case opcodes.GetItem():
                arg = self.stack.pop()
                self.stack.push(self.stack.pop()[arg])
            case opcodes.PushBound():
                self.stack.push_bound()
            case opcodes.PopBound():
                self.stack.pop_bound()
            case opcodes.DiscardBound():
                self.stack.discard_bound()
            case opcodes.SetTop(arg):
                self.stack.set_top(arg)
            case opcodes.Call(arg):
                args = []
                while self.stack.get_top() > 0:
                    args.insert(0, self.stack.pop())
                self.stack.pop_bound()
                func = self.stack.pop(expect={"function"})
                lim = None if arg == 0 else arg - 1
                if isinstance(func, ltypes.Function):
                    self.thread.call_stack.append(Frame(
                        self.thread,
                        *func.bytecode, func.offs,
                        func.upvalues, args[len(func.argnames):],
                        lim, False,
                        [Block(None, (0, 0), dict(zip(func.argnames, args + [ltypes.nil] * (len(func.argnames) - len(args)))))]
                    ))
                else:  # BuiltinFunction
                    self.thread.call_stack.append(ForeignFrame(self.thread, lim, func.f(*args)))
            case opcodes.Negative():
                self.stack.push(ltypes.Number(-self.stack.pop(expect={"number"}).f))
            case opcodes.Sign():
                arg = self.stack.pop(expect={"number"})
                self.stack.push(ltypes.Number(0.0 if arg == 0 else [-1.0, 1.0][arg.f > 0]))
            case opcodes.Len():
                arg = self.stack.pop(expect={"table", "string"})
                self.stack.push(arg.getn())
            case opcodes.Equal():
                arg2, arg1 = self.stack.pop(), self.stack.pop()
                self.stack.push(ltypes.true if arg1 == arg2 else ltypes.false)
            case opcodes.Unequal():
                arg2, arg1 = self.stack.pop(), self.stack.pop()
                self.stack.push(ltypes.true if arg1 != arg2 else ltypes.false)
            case opcodes.Concat():
                arg2, arg1 = self.stack.pop(expect={"number", "string"}), self.stack.pop(expect={"number", "string"})
                self.stack.push(ltypes.String(arg1.tostring().s + arg2.tostring().s))
            case opcodes.Less():
                arg = self.stack.pop(expect={"number"})
                self.stack.push(ltypes.true if self.stack.pop(expect={"number"}).f < arg.f else ltypes.false)
            case opcodes.Greater():
                arg = self.stack.pop(expect={"number"})
                self.stack.push(ltypes.true if self.stack.pop(expect={"number"}).f > arg.f else ltypes.false)
            case opcodes.LessEqual():
                arg = self.stack.pop(expect={"number"})
                self.stack.push(ltypes.true if self.stack.pop(expect={"number"}).f <= arg.f else ltypes.false)
            case opcodes.GreaterEqual():
                arg = self.stack.pop(expect={"number"})
                self.stack.push(ltypes.true if self.stack.pop(expect={"number"}).f >= arg.f else ltypes.false)
            case opcodes.Add():
                self.stack.push(ltypes.Number(self.stack.pop(expect={"number"}).f + self.stack.pop(expect={"number"}).f))
            case opcodes.Subtract():
                self.stack.push(ltypes.Number(-self.stack.pop(expect={"number"}).f + self.stack.pop(expect={"number"}).f))
            case opcodes.Multiply():
                self.stack.push(ltypes.Number(self.stack.pop(expect={"number"}).f * self.stack.pop(expect={"number"}).f))
            case opcodes.Divide():
                arg = self.stack.pop()
                self.stack.push(ltypes.Number(self.stack.pop(expect={"number"}).f / arg.f))
            case opcodes.SetDual(arg):
                values = [self.stack.pop() for _ in range(arg)]
                for what in values:
                    ref = self.stack.pop(allow_ref=True)
                    if not isinstance(ref, ltypes.Ref):
                        raise ltypes.InterpreterError("assign to rvalue (how??)")
                    ref.set(what)
            case opcodes.Return(arg):
                if arg == 0:
                    results = []
                    while self.stack.get_top() > 0:
                        results.insert(0, self.stack.pop())
                    self.stack.pop_bound()
                else:
                    results = [self.stack.pop() for _ in range(arg)]
                lim = self.return_limit
                self.thread.call_stack.pop()
                if lim is None:
                    for i in results:
                        self.thread.call_stack[-1].stack.push(i)
                else:
                    for i, _ in zip(results, range(lim)):
                        self.thread.call_stack[-1].stack.push(i)
            case opcodes.BlockInfo(arg):
                block_locals = self.block_stack[-1].locals
                for idx in zip(arg[::4], arg[1::4], arg[2::4], arg[3::4]):
                    string = self.strings[int(bytes(idx))]
                    if string not in block_locals:
                        block_locals[string] = ltypes.nil
            case opcodes.PushBlock(arg):
                self.block_stack.append(Block(None if arg == 0 else arg, (len(self.stack.stacks), self.stack.get_top()), {}))
            case opcodes.PopBlock():
                block = self.block_stack.pop()
                if len(self.stack.stacks) >= block.stack_level[0]:
                    del self.stack.stacks[block.stack_level[0]+1:]
                else:
                    raise ltypes.InterpreterError("stack went under block depth? how")
                self.stack.set_top(block.stack_level[1])
            case opcodes.Break():
                while True:
                    if not self.block_stack:
                        raise ltypes.InterpreterError("nothing to break out of")
                    block = self.block_stack.pop()
                    if block.break_addr is not None:
                        self.ip = block.break_addr - 1  # TODO: -1?
                        break
            case opcodes.Swap():
                v1 = self.stack.pop(allow_ref=True)
                v2 = self.stack.pop(allow_ref=True)
                self.stack.push(v1)
                self.stack.push(v2)
            # TODO: bool coercion?
            case opcodes.If():
                block1 = self.stack.pop()
                cond = self.stack.pop(expect={"boolean"})
                if cond.b:
                    self.ip = block1.offs - 1
            case opcodes.IfElse():
                block2 = self.stack.pop()
                block1 = self.stack.pop()
                cond = self.stack.pop(expect={"boolean"})
                if cond.b:
                    self.ip = block1.offs - 1
                else:
                    self.ip = block2.offs - 1
            case opcodes.Jump(arg):
                self.ip = arg - 1
            case opcodes.Dup(arg):
                self.stack.push(self.stack.peek(arg, cross_boundary=True))
            case opcodes.Nop(arg) | opcodes.UpvalueInfo(arg):
                pass
            case _:
                raise NotImplementedError(f"{opcode}")
        self.ip += 1

PopThread = object()

class Thread:
    call_stack: list[Frame]
    global_env: ltypes.Table
    last_input: tuple[ltypes.LuaType, ...] | list[bytes]
    last_output: tuple[ltypes.LuaType, ...]

    def __init__(self, env):
        # We start with a single dummy frame for collecting return values and the such
        self.call_stack = [Frame(self, [], [], 0, {}, [], None, False, [])]
        self.global_env = env
        self.alive = True
        self.last_input = []
        self.last_output = ()

    def tick(self):
        if not self.alive:
            raise ltypes.InterpreterError("coroutine.resume(...) dead thread")
        if len(self.call_stack) == 1:
            self.alive = False
            self.last_output = tuple(reversed([self.call_stack[0].stack.pop() for _ in range(self.call_stack[0].stack.get_top())]))
            return PopThread
        try:
            return self.call_stack[-1].tick()
        except ltypes.InterpreterError as e:
            self.call_stack.pop()
            return self.throw(e)

    def throw(self, err):
        while len(self.call_stack) > 1:
            if isinstance(self.call_stack[-1], ForeignFrame):
                # Throw it inside the ForeignFrame and continue execution from there (no propagating errors across Python functions)
                self.call_stack[-1].to_throw = err
                return None
            for block in self.call_stack[-1].block_stack:
                if b"_ERR" in block.locals:
                    block.locals[b"_ERR"] = ltypes.String(str(err).encode("utf8"))
                    return None
            self.call_stack.pop()
        self.alive = False
        self.last_output = tuple(reversed([self.call_stack[0].stack.pop() for _ in range(self.call_stack[0].stack.get_top())]))
        raise err

    def call(self, func, args):
        assert isinstance(self.call_stack[-1], ForeignFrame)  # We should only be able to invoke call(...) while evaluating a BuiltinFunction
        if isinstance(func, ltypes.BuiltinFunction):
            self.call_stack.append(ForeignFrame(self, None, func.f(*args)))
        else:
            self.call_stack.append(Frame(
                self,
                *func.bytecode, func.offs,
                func.upvalues, [],
                None, False,
                [Block(None, (0, 0), dict(zip(func.argnames, args + [ltypes.nil] * (len(func.argnames) - len(args)))))]
            ))
        # Return to the interpreter loop
        yield
        results = []
        while self.call_stack[-1].stack.get_top():
            results.insert(0, self.call_stack[-1].stack.pop())
        self.call_stack[-1].stack.set_top(0)
        return results

class Interpreter:
    thread_stack: list[Thread]
    global_env: ltypes.Table

    def __init__(self, env):
        self.thread_stack = []
        self.global_env = env
        self.global_env[ltypes.String(b"_G")] = self.global_env

    def call(self, func, args):
        return (yield from self.thread_stack[-1].call(func, args))

    def mainloop(self, entry):
        if self.thread_stack:
            raise ltypes.InterpreterError("This interpreter is already running")
        def _entry():
            yield from self.thread_stack[0].call(entry, [])
            return ()
        self.thread_stack.append(Thread(self.global_env))
        if isinstance(entry, ltypes.BuiltinFunction):
            self.thread_stack[0].call_stack.append(ForeignFrame(self.thread_stack[0], None, _entry()))
        else:
            self.thread_stack[0].call_stack.append(ForeignFrame(self.thread_stack[0], None, entry))
        while self.thread_stack:
            try:
                res = self.thread_stack[-1].tick()
            except ltypes.InterpreterError as err:
                self.thread_stack.pop()
                if not self.thread_stack:
                    raise
                res = self.thread_stack[-1].throw(err)
            if res is PopThread:
                self.thread_stack.pop()
            elif isinstance(res, Thread):
                self.thread_stack.append(res)
            elif res is not None:
                raise RuntimeError(f"tick(...) returned invalid {res}")
