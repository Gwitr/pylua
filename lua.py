# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

from contextlib import contextmanager

import opcodes
from llexer import TokenStream, ParseError
from lparser import InstructionStream, parse_chunk

DEBUG = False

bound = object()

class Interpreter():

    def __init__(self, env):
        self.ip = 0
        self.stack = []
        self.call_stack = []
        self.global_env = env
        self.global_env["_G"] = self.global_env

    def pop(self, expect=None):
        # TODO: Reimplement source info (local/upvalue/global)
        if not self.stack:
            raise RuntimeError("pop off empty stack")
        result = self.stack.pop()
        if result is bound:
            raise RuntimeError("pop across bound")
        if isinstance(result, list):
            raise RuntimeError("pop reference")
        if expect is not None and self.lua_type(result) not in expect:
            raise RuntimeError(f"expected {', '.join(expect)}, got {self.lua_type(result)}")
        return result

    def lua_type(self, value):
        if value is None:
            return "nil"
        return {
            str: "string", float: "number", bool: "boolean", dict: "table", tuple: "function"
        }.get(type(value), f"unknown({type(value).__qualname__})")

    def get_top(self):
        try:
            return self.stack[::-1].index(bound)
        except ValueError:
            return len(self.stack)

    def repr_value(self, value):
        if value is None:
            return "nil"
        match value:
            case str(x) | float(x):
                return repr(x)
            case bool(x):
                return repr(x).lower()
            case dict(x):
                return "{" + ", ".join(
                    f"{k}={self.repr_value(v)}" if isinstance(k, str) else f"[{self.repr_value(k)}]={self.repr_value(v)}"
                    for k, v in x.items()) + "}"
            case tuple(x):
                return f"func({', '.join(x[0])})"
            case int(x):
                return f"code at {x}"
            case [idx, table, _del_on_nil]:
                table_repr = self.repr_value(table)
                if len(table_repr) > 20:
                    table_repr = table_repr[:17] + "..."
                return f"{table_repr}[{idx!r}]"
            case x if x is bound:
                return "|"
        return f"??? {value}"

    def run(self, call_stack_level):
        self.ip, _, _, _, bytecode = self.call_stack[-1]
        strings, ops = opcodes.decode_bytecode(bytecode)
        while True:
            if call_stack_level is not None and len(self.call_stack) <= call_stack_level:
                break
            if call_stack_level is None and self.ip >= len(ops):
                break
            opcode, offs = ops[self.ip]
            if DEBUG:
                call_stack_debug = "; ".join(
                    f"[{','.join(localvars.keys())}]"
                    if addr == -1 else
                    f"[{','.join(localvars.keys())}],[{','.join(upvalues.keys())}]"
                    for addr, localvars, upvalues, _, _ in self.call_stack
                )
                stack_debug = " ".join(self.repr_value(v) for v in self.stack)
                print(f"{self.ip:04x}{' '*len(self.call_stack)+str(opcode):<50} | {stack_debug:<100} | {call_stack_debug}")

            match opcode:
                case opcodes.PushString(arg):
                    self.stack.append(arg)
                case opcodes.PushNumber(arg):
                    self.stack.append(float(arg))
                case opcodes.PushNil():
                    self.stack.append(None)
                case opcodes.PushTrue():
                    self.stack.append(True)
                case opcodes.PushFalse():
                    self.stack.append(False)
                case opcodes.PushInline(arg):
                    self.stack.append(self.ip + 1)
                    self.ip += arg
                case opcodes.PushFunction(arg):
                    # NOTE: This assumes that the first instruction of a function is always UpvalueInfo. That's kind of ugly
                    argnames = reversed([self.pop() for _ in range(arg)])
                    offs = self.pop()
                    upvalues_string = ops[offs][0].arg
                    upvalues = {}
                    for idx in zip(upvalues_string[::4], upvalues_string[1::4], upvalues_string[2::4], upvalues_string[3::4]):
                        upvalue_name = strings[int("".join(idx))]
                        for _, scope, scope_upvalues, _, _ in self.call_stack[::-1]:
                            if upvalue_name in scope_upvalues:
                                upvalues[upvalue_name] = scope_upvalues[upvalue_name]
                            elif upvalue_name in scope:
                                upvalues[upvalue_name] = scope
                    self.stack.append((tuple(argnames), offs, bytecode, upvalues))
                case opcodes.PushTable(arg):
                    trailing_values = []
                    while self.get_top() > 0:
                        trailing_values.insert(0, self.pop())
                    self.stack.pop()
                    # TODO: Is this behavior 100% accurate
                    arr = []
                    for _ in range(arg):
                        v = self.pop()
                        k = self.pop()
                        arr.append((k, v))
                    next_index = 1
                    table = {}
                    for k, v in reversed(arr):
                        if k is None:
                            k = float(next_index)
                            next_index += 1
                        table[k] = v
                    for i, v in enumerate(trailing_values, next_index):
                        table[float(i)] = v
                    self.stack.append(table)
                case opcodes.RefName(arg):
                    _, localvars, upvalues, _, _ = self.call_stack[-1]
                    if arg in localvars:
                        self.stack.append([arg, localvars, False])
                    elif arg in upvalues:
                        self.stack.append([arg, upvalues[arg], False])
                    else:
                        self.stack.append([arg, upvalues["_ENV"]["_ENV"], False])
                case opcodes.GetName(arg):
                    _, localvars, upvalues, _, _ = self.call_stack[-1]
                    if arg in localvars:
                        self.stack.append(localvars[arg])
                    elif arg in upvalues:
                        self.stack.append(upvalues[arg][arg])
                    else:
                        self.stack.append(upvalues["_ENV"]["_ENV"].get(arg))
                case opcodes.RefItem():
                    self.stack.append([self.pop(), self.pop(), True])
                case opcodes.GetItem():
                    arg = self.pop()
                    self.stack.append(self.pop().get(arg, None))
                case opcodes.PushBound():
                    self.stack.append(bound)
                case opcodes.PopBound():
                    self.pop_bound()
                case opcodes.DiscardBound():
                    self.discard_bound()
                case opcodes.SetTop(arg):
                    self.set_top(arg)
                case opcodes.Call(arg):
                    args = []
                    while self.get_top() > 0:
                        args.insert(0, self.pop())
                    self.stack.pop()
                    func = self.pop(expect={"function"})
                    if isinstance(func[1], int):
                        lim = None if arg == 0 else arg - 1
                        self.call_stack.append([self.ip, dict(zip(func[0], args + [None] * (len(func[0]) - len(args)))), func[3], lim, func[2]])
                        self.stack.append(bound)
                        bytecode = self.call_stack[-1][4]
                        strings, ops = opcodes.decode_bytecode(bytecode)
                        self.ip = func[1] - 1
                    else:
                        try:
                            if func[1](self, args) is not None:
                                raise RuntimeError("can't yield past Interpreter.call")
                        except StopIteration:
                            pass
                case opcodes.Negative():
                    self.stack.append(-self.pop(expect={"number"}))
                case opcodes.Sign():
                    arg = self.pop(expect={"number"})
                    self.stack.append(0.0 if arg == 0 else [-1.0, 1.0][arg > 0])
                case opcodes.Len():
                    arg = self.pop(expect={"table", "string"})
                    if isinstance(arg, str):
                        self.stack.append(float(len(arg)))
                    elif isinstance(arg, dict):
                        i = 1
                        while i in arg:
                            i += 1
                        self.stack.append(float(i - 1))
                case opcodes.Equal():
                    arg2, arg1 = self.pop(), self.pop()
                    self.stack.append(type(arg1) is type(arg2) and (
                        (isinstance(arg1, (tuple, dict)) and arg1 is arg2) or
                        (isinstance(arg1, (float, bool, str)) and arg1 == arg2) or
                        (arg1 is None and arg2 is None)
                    ))
                case opcodes.Unequal():
                    arg2, arg1 = self.pop(), self.pop()
                    self.stack.append(not (type(arg1) is type(arg2) and (
                        (isinstance(arg1, (tuple, dict)) and arg1 is arg2) or
                        (isinstance(arg1, (float, bool, str)) and arg1 == arg2) or
                        (arg1 is None and arg2 is None)
                    )))
                case opcodes.Concat():
                    arg2, arg1 = self.pop(expect={"number", "string"}), self.pop(expect={"number", "string"})
                    self.stack.append(str(arg1) + str(arg2))
                case opcodes.Less():
                    arg = self.pop(expect={"number"})
                    self.stack.append(self.pop(expect={"number"}) < arg)
                case opcodes.Greater():
                    arg = self.pop(expect={"number"})
                    self.stack.append(self.pop(expect={"number"}) > arg)
                case opcodes.LessEqual():
                    arg = self.pop(expect={"number"})
                    self.stack.append(self.pop(expect={"number"}) <= arg)
                case opcodes.GreaterEqual():
                    arg = self.pop(expect={"number"})
                    self.stack.append(self.pop(expect={"number"}) >= arg)
                case opcodes.Add():
                    self.stack.append(self.pop(expect={"number"}) + self.pop(expect={"number"}))
                case opcodes.Subtract():
                    self.stack.append(-self.pop(expect={"number"}) + self.pop(expect={"number"}))
                case opcodes.Multiply():
                    self.stack.append(self.pop(expect={"number"}) * self.pop(expect={"number"}))
                case opcodes.Divide():
                    arg = self.pop()
                    self.stack.append(self.pop(expect={"number"}) / arg)
                case opcodes.SetDual(arg):
                    values = [self.pop() for _ in range(arg)]
                    for what in values:
                        value = self.stack.pop()
                        if value is bound:
                            raise RuntimeError("pop across bound")
                        if not isinstance(value, list):
                            raise RuntimeError("assign to rvalue (how??)")
                        if what is None and value[2]:
                            del value[1][value[0]]
                        else:
                            value[1][value[0]] = what
                case opcodes.Return(arg):
                    if arg == 0:
                        results = []
                        while self.get_top() > 0:
                            results.insert(0, self.pop())
                        self.stack.pop()
                    else:
                        results = [self.pop() for _ in range(arg)]
                    while self.stack.pop() is not bound:
                        pass
                    self.ip, _locals, _upvars, max_return, bytecode = self.call_stack.pop()
                    while self.ip == -1:
                        # pop all blocks
                        self.ip, _locals, _upvars, max_return, bytecode = self.call_stack.pop()
                    if max_return is None:
                        for i, in zip(results):
                            self.stack.append(i)
                    else:
                        for i, _ in zip(results, range(max_return)):
                            self.stack.append(i)
                    strings, ops = opcodes.decode_bytecode(bytecode)
                case opcodes.BlockInfo(arg):
                    for idx in zip(arg[::4], arg[1::4], arg[2::4], arg[3::4]):
                        self.call_stack[-1][1].setdefault(strings[int("".join(idx))], None)
                case opcodes.PushBlock(arg):
                    break_addr = None if arg == 0 else arg
                    self.call_stack.append([-1, {}, self.call_stack[-1][2] | {k: self.call_stack[-1][1] for k in self.call_stack[-1][1]}, break_addr, bytecode])
                case opcodes.PopBlock():
                    if self.call_stack[-1][0] != -1:
                        raise RuntimeError("pop function block")
                    self.call_stack.pop()
                case opcodes.Break():
                    while True:
                        addr, _localvars, _upvalues, retlimit_or_break_addr = self.call_stack.pop()
                        if addr != -1 or len(self.call_stack) == 0:
                            raise RuntimeError("Nothing to break from")
                        if retlimit_or_break_addr is not None:
                            self.ip = retlimit_or_break_addr
                            break
                case opcodes.Swap():
                    v1 = self.stack.pop()
                    v2 = self.stack.pop()
                    self.stack.append(v1)
                    self.stack.append(v2)
                # TODO: bool coercion?
                case opcodes.If():
                    block1 = self.pop()
                    cond = self.pop(expect={"boolean"})
                    if cond:
                        self.ip = block1 - 1
                case opcodes.IfElse():
                    block2 = self.pop()
                    block1 = self.pop()
                    cond = self.pop(expect={"boolean"})
                    if cond:
                        self.ip = block1 - 1
                    else:
                        self.ip = block2 - 1
                case opcodes.Jump(arg):
                    self.ip = arg - 1
                case opcodes.Dup(arg):
                    self.stack.append(self.stack[-arg-1])
                case opcodes.Nop(arg) | opcodes.UpvalueInfo(arg):
                    pass
                case _:
                    raise NotImplementedError(f"{opcode}")
            self.ip += 1

    def set_top(self, arg):
        if arg < 0:
            pos = len(self.stack) - 1 - self.stack[::-1].index(bound)
            top = len(self.stack) - pos - 1
            arg = top + arg
        while True:
            pos = len(self.stack) - 1 - self.stack[::-1].index(bound)
            top = len(self.stack) - pos - 1
            if top == arg:
                break
            if top > arg:
                self.stack.pop()
            else:
                self.stack.append(None)

    def discard_bound(self):
        del self.stack[len(self.stack) - 1 - self.stack[::-1].index(bound)]

    def pop_bound(self):
        del self.stack[len(self.stack) - 1 - self.stack[::-1].index(bound):]

    @contextmanager
    def push_stack_guard(self, top=0):
        self.stack.append(bound)
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

    def call(self, func, args):
        self.stack.append(bound)
        self.call_stack.append([func[1], dict(zip(func[0], args + [None] * (len(func[0]) - len(args)))), func[3], None, func[2]])
        self.stack.append(bound)
        self.run(len(self.call_stack) - 1)
        results = []
        while self.stack[-1] != bound:
            results.insert(0, self.pop())
        self.stack.pop()
        return results

def lua_tostring(interpreter, args):
    if len(args) != 1:
        raise RuntimeError("tostring expects 1 parameter")
    if isinstance(args[0], tuple):
        interpreter.stack.append(f"function: 0x{id(args[0]):08x}")
    elif isinstance(args[0], dict):
        interpreter.stack.append(f"table: 0x{id(args[0]):08x}")
    elif isinstance(args[0], bool):
        interpreter.stack.append("true" if args[0] else "false")
    elif isinstance(args[0], str):
        interpreter.stack.append(args[0])
    elif isinstance(args[0], float):
        interpreter.stack.append(str(args[0]))
    elif args[0] is None:
        interpreter.stack.append("nil")
    else:
        interpreter.stack.append(f"unknown({type(args[0]).__qualname__}): 0x{id(args[0]):08x}")

def lua_print(interpreter, args):
    to_print = []
    for arg in args:
        lua_tostring(interpreter, (arg,))
        to_print.append(interpreter.pop())
    print(*to_print)
    interpreter.stack.append(None)

def lua_type(interpreter, args):
    if len(args) != 1:
        raise RuntimeError("type expects 1 parameter")
    return interpreter.lua_type(args[0])

def load(interpreter, chunk, chunkname="<string>", mode="bt", env=None):
    if isinstance(chunk, tuple):
        s = ""
        while True:
            success, *results = yield from interpreter.pcall(chunk, [])
            if not success:
                return None, results[0]
            if (not results) or results[0] is None or results[0] == "":
                break
            s += results[0]
        chunk = s

    if mode == "bt":
        mode = "b" if chunk[0] == "$" else "t"

    if mode == "t":
        out = InstructionStream()
        try:
            parse_chunk(TokenStream(chunk, chunkname), out)
        except ParseError as e:
            return None, str(e)
        if DEBUG:
            with open(f"bytecode-{chunkname}.txt", "w", encoding="ascii") as f:
                strings, ops = opcodes.decode_bytecode(out.serialize())
                print(strings, file=f)
                inline_countdowns = []
                indent = 0
                for ip, (instr, _) in enumerate(ops):
                    print(f"{ip:03x} | {'  '*indent}{str(instr)}", file=f)
                    if inline_countdowns:
                        inline_countdowns = [i - 1 for i in inline_countdowns]
                        indent -= inline_countdowns.count(0)
                        inline_countdowns = [i for i in inline_countdowns if i != 0]
                    if isinstance(instr, opcodes.PushInline):
                        indent += 1
                        inline_countdowns.append(instr.arg)
        chunk = out.serialize()

    with interpreter.push_stack_guard(top=1):
        interpreter.call_stack.append([0, {"_ENV": env if env else interpreter.global_env}, {}, None, chunk])
        interpreter.run(None)
        interpreter.call_stack.pop()

    return (interpreter.stack.pop(),)

def lua_load(interpreter, *args):
    if len(args) not in range(1, 4):
        raise RuntimeError("load expects 1-4 parameters")
    interpreter.stack += yield from load(interpreter, *args)

def lua_bor(interpreter, args):
    """Temporary function for until I stop being too lazy to implement the short-circuiting or operator"""
    if len(args) < 1:
        raise RuntimeError("bor expects 1+ parameters")
    val = None
    for val in args:
        # TODO: pcall here
        if isinstance(val, tuple):
            val = interpreter.call(val, [])
        if isinstance(val, (dict, tuple)) or (isinstance(val, float) and val != 0.0) or (isinstance(val, str) and val != "") or val is True:
            interpreter.stack.append(val)
            return
    interpreter.stack.append(val)
    return

def main():
    interp = Interpreter({
        "print": ((), lua_print, {}),
        "tostring": ((), lua_tostring, {}),
        "type": ((), lua_type, {}),
        "lua_load": ((), lua_load, {}),
        "bor": ((), lua_bor, {})
    })
    func, rest = None, None
    with open((filename := "table.lua"), encoding="ascii") as f:
        try:
            load_gen = load(interp, f.read(), filename)
            while True:
                next(load_gen)
        except StopIteration as e:
            func, *rest = e.value
    if func is None:
        print("syntax error:", rest[0])
    else:
        interp.call(func, [])
        print(interp.stack)

if __name__ == "__main__":
    main()
