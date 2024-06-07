# pylint: disable=missing-function-docstring

"""Contains facilities to load the standard library, and definitions for all foreign builtin functions"""

import os
import re
import sys
import math
import random

import ltypes
import linterp
import opcodes
from lparser import parse
from lgrammar import grammar
from llexer import Tokens, ParseError

DEBUG = False

def load(interp):
    """Loads the Lua standard library"""
    # Load the foreign functions first
    for name, func in FUNCTIONS.items():
        interp.global_env[name] = ltypes.BuiltinFunction(interp, func)
    for mname, functions in MODULES.items():
        interp.global_env[mname] = ltypes.Table()
        for name, func in functions.items():
            interp.global_env[mname][name] = ltypes.BuiltinFunction(interp, func)
    # Then, load the Lua side of the standard library
    def entry():
        builtin = ["basic.lua", "table.lua", "coroutine.lua", "math.lua"]
        func, rest = None, None
        for filename in builtin:
            with open(os.path.join(os.path.dirname(__file__), filename), encoding="ascii") as f:
                func, *rest = yield from interp.call(interp.global_env[ltypes.String(b"load")], [f.read()])
            if func is None:
                raise ltypes.InterpreterError("syntax error: " + rest[0])
            yield from interp.call(func, [])
        return ()
    interp.mainloop(entry())

FUNCTIONS = {}
MODULES = {}

def register_global_function(func):
    FUNCTIONS[ltypes.String(re.sub("^lua_", "", func.__name__).encode("ascii"))] = func
    return func

def register_module_function(module_name):
    def decorator(func):
        fname_encoded = ltypes.String(re.sub(f"^lua_{module_name}_", "", func.__name__).encode("ascii"))
        MODULES.setdefault(ltypes.String(module_name.encode("ascii")), {})[fname_encoded] = func
        return func
    return decorator

def check_type(paramn, value, types):
    if not isinstance(types, tuple):
        types = (types,)
    if not isinstance(value, types):
        raise ltypes.InterpreterError(f"parameter #{paramn}: expected {', '.join({t.TYPE for t in types})}, got {value.TYPE}")

def require_params(minp, maxp=None):
    """Helper for functions that take a specific amount of parameters"""
    if maxp is None:
        maxp = minp
    def decorator(func):
        funcname = re.sub("^lua_", "", func.__name__)
        def wrap(interpreter, *args):
            if maxp == float("inf"):
                if len(args) < minp:
                    raise ltypes.InterpreterError(f"{funcname} expects {minp}+ parameters, got {len(args)}")
                return func(interpreter, *args)
            if len(args) not in range(minp, maxp+1):
                plural = minp != maxp
                raise ltypes.InterpreterError(f"{funcname} expects {minp}{("-"+str(maxp))*plural} parameter{'s'*plural}, got {len(args)}")
            return func(interpreter, *args)
        wrap.__name__ = func.__name__
        wrap.__qualname__ = func.__qualname__
        return wrap
    return decorator

### Standard library code begins here

## basic

@register_global_function
@require_params(1, 2)
def lua_next(_interpreter, table, key=ltypes.nil):
    if not isinstance(table, ltypes.Table):
        raise ltypes.InterpreterError(f"expected table, got {table.type.s.decode('ascii')}")
    key = table.nextkey(key)
    return key, table[key]
    yield

@register_global_function
@require_params(1)
def lua_tostring(_interpreter, value):
    return value.tostring()
    yield

@register_global_function
def lua_print(_interpreter, *args):
    to_print = []
    for arg in args:
        to_print.append(arg.tostring().s.decode("utf8", errors="ignore"))
    print(*to_print)
    return ()
    yield

@register_global_function
@require_params(1)
def lua_type(_interpreter, value):
    return value.type
    yield

@register_global_function
@require_params(1, 4)
def lua_load(interpreter, chunk, chunkname=ltypes.String(b"<string>"), mode=ltypes.String(b"bt"), env=ltypes.nil):
    chunkname = chunkname.tostring().s.decode('ascii', errors='backslashreplace')
    mode = mode.tostring().s
    if isinstance(chunk, (ltypes.Function, ltypes.BuiltinFunction)):
        s = b""
        while True:
            success, *results = yield from interpreter.call(chunk, [])
            if not success:
                return ltypes.nil, results[0]
            if (not results) or results[0] is None or results[0] == "":
                break
            s += results[0].tostring().s
        chunk = s

    if {*mode} == {*b"bt"}:
        mode = b"b" if chunk[0] == b"$" else b"t"

    if mode == b"t":
        try:
            chunk = parse(grammar, Tokens.from_string(chunk, chunkname))
        except ParseError as e:
            return ltypes.nil, ltypes.String(str(e).encode("utf8"))
        if DEBUG:
            with open(f"bytecode-{chunkname}.txt", "w", encoding="ascii") as f:
                strings, ops = opcodes.decode(chunk)
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

    func = ltypes.Function(interpreter, (), 0, opcodes.decode(chunk), {b"_ENV": {b"_ENV": env if env != ltypes.nil else interpreter.global_env}})
    return (func,)

@register_global_function
def lua_bor(interpreter, *args):
    """Temporary function for until I stop being too lazy to implement the short-circuiting or operator"""
    val = ltypes.nil
    for val in args:
        if isinstance(val, ltypes.Function):
            val, = yield from interpreter.call(val, [])
        if val is ltypes.true:
            return val
        if (isinstance(val, ltypes.Number) and val.f != 0.0) or (isinstance(val.f, ltypes.String) and val.s != b""):
            return val
        if isinstance(val, (ltypes.Function, ltypes.BuiltinFunction, ltypes.Table)):
            return val
    return val

@register_global_function
def lua_error(_interpreter, *args):
    # TODO: level parameter, tracebacks, etc
    arg = args[0] if args else ltypes.nil
    raise ltypes.InterpreterError(arg.tostring().unicode())
    yield  # pylint: disable=unreachable

@register_global_function
def lua_tonumber(_interpreter, args):
    if len(args) not in range(1, 3):
        raise ltypes.InterpreterError("tonumber expects 1-2 parameters")
    if len(args) > 1:
        # TODO: implement this
        raise ltypes.InterpreterError("tonumber non-decimal bases not implemented yet")
    e = args[0]
    if isinstance(e, ltypes.Number):
        return e
    if isinstance(e, ltypes.String):
        try:
            return ltypes.Number(float(e.s))
        except ValueError:
            return ltypes.nil
    return ltypes.nil
    yield

## coroutine

@register_module_function("coroutine")
@require_params(0)
def lua_coroutine_running(interpreter):
    return ltypes.ThreadContainer(interpreter.thread_stack[-1]), (ltypes.true if len(interpreter.thread_stack) == 1 else ltypes.false)
    yield

@register_module_function("coroutine")
@require_params(1)
def lua_coroutine_create(interpreter, function):
    thread = linterp.Thread(interpreter.global_env)

    if isinstance(function, ltypes.BuiltinFunction):
        raise NotImplementedError  # TODO: Implement this

    elif isinstance(function, ltypes.Function):
        thread.call_stack.append(linterp.Frame(
            thread,
            *function.bytecode, function.offs,
            function.upvalues, [],
            None, False,
            [linterp.Block(None, (0, 0), {})]
        ))

    else:
        raise ltypes.InterpreterError(f"expected function, got {function.type.s.decode('ascii')}")

    thread.last_input = list(function.argnames)
    return ltypes.ThreadContainer(thread)
    yield

@register_module_function("coroutine")
@require_params(1, float("inf"))
def lua_coroutine_resume(_interpreter, thread, *args):
    if not isinstance(thread, ltypes.ThreadContainer):
        raise ltypes.InterpreterError(f"expected thread, got {thread.type.s.decode('ascii')}")
    if not thread.thread.alive:
        return ltypes.false, ltypes.String(b"cannot resume dead coroutine")
    if isinstance(thread.thread.last_input, list):
        argnames = thread.thread.last_input
        thread.thread.last_input = ()
        thread.thread.call_stack[-1].block_stack[0].locals.update(zip(argnames, list(args) + [ltypes.nil] * (len(argnames) - len(args))))
    else:
        thread.thread.last_input = args
    try:
        yield thread.thread
    except ltypes.InterpreterError as e:
        return ltypes.false, ltypes.String(str(e).encode("utf8"))
    return ltypes.true, *thread.thread.last_output

@register_module_function("coroutine")
def lua_coroutine_yield(interpreter, *args):
    interpreter.thread_stack[-1].last_output = args
    yield linterp.PopThread
    return interpreter.thread_stack[-1].last_input

@register_module_function("coroutine")
@require_params(1)
def lua_coroutine_status(interpreter, coro):
    if not isinstance(coro, ltypes.ThreadContainer):
        raise ltypes.InterpreterError(f"expected thread, got {coro.type.s.decode('ascii')}")
    if coro.thread is interpreter.thread_stack[-1]:
        return ltypes.String(b"running")
    if coro.thread in interpreter.thread_stack:
        return ltypes.String(b"normal")
    if coro.thread.alive:
        return ltypes.String(b"suspended")
    return ltypes.String(b"dead")
    yield

## math

# TODO: Type coercion for all of these!

@register_module_function("math")
@require_params(1)
def lua_math_abs(_, x):
    check_type(1, x, (ltypes.Number,))
    return ltypes.Number(abs(x.f))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_acos(_, x):
    check_type(1, x, (ltypes.Number,))
    try:
        return ltypes.Number(math.acos(x.f))
    except ValueError:
        return ltypes.Number(float("nan"))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_asin(_, x):
    check_type(1, x, (ltypes.Number,))
    try:
        return ltypes.Number(math.asin(x.f))
    except ValueError:
        return ltypes.Number(float("nan"))
    yield

@register_module_function("math")
@require_params(1, 2)
def lua_math_atan2(_, y, x=ltypes.Number(1.0)):
    check_type(1, y, (ltypes.Number,))
    check_type(2, x, (ltypes.Number,))
    try:
        return ltypes.Number(math.atan2(y.f, x.f))
    except ValueError:
        return ltypes.Number(float("nan"))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_ceil(_, x):
    check_type(1, x, (ltypes.Number,))
    return ltypes.Number(math.ceil(x.f))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_cos(_, x):
    check_type(1, x, (ltypes.Number,))
    return ltypes.Number(math.cos(x.f))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_deg(_, x):
    check_type(1, x, (ltypes.Number,))
    return ltypes.Number(math.degrees(x.f))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_exp(_, x):
    check_type(1, x, (ltypes.Number,))
    try:
        return ltypes.Number(math.exp(x.f))
    except ValueError:
        return ltypes.Number(float("nan"))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_floor(_, x):
    check_type(1, x, (ltypes.Number,))
    return ltypes.Number(math.floor(x.f))
    yield

@register_module_function("math")
@require_params(2)
def lua_math_fmod(_, x, y):
    check_type(1, x, (ltypes.Number,))
    check_type(2, y, (ltypes.Number,))
    if y.f == 0:
        return ltypes.Number(float("-nan"))
    if (x.f < 0 and y.f < 0) or (x.f > 0 and y.f > 0):
        return ltypes.Number(x.f % y.f)
    else:
        return ltypes.Number(-(x.f % y.f))
    yield

@register_module_function("math")
@require_params(0)
def lua_math__huge(_):
    return ltypes.Number(sys.float_info.max)
    yield

@register_module_function("math")
@require_params(2)
def lua_math_log(_, x, base=ltypes.Number(math.e)):
    check_type(1, x, (ltypes.Number,))
    check_type(2, base, (ltypes.Number,))
    try:
        return ltypes.Number(math.log(x.f, base.f))
    except ValueError:
        return ltypes.Number(float("nan"))
    yield

@register_module_function("math")
@require_params(1, float("inf"))
def lua_math_max(_, x, *rest):
    check_type(1, x, (ltypes.Number,))
    maxarg = x
    for i, arg in enumerate(rest, 2):
        check_type(i, arg, (ltypes.Number,))
        if maxarg.f < arg.f:
            maxarg = arg
    return maxarg
    yield

@register_module_function("math")
@require_params(0)
def lua_math__maxinteger(_):
    return ltypes.Number(float(sys.maxsize))
    yield

@register_module_function("math")
@require_params(1, float("inf"))
def lua_math_min(_, x, *rest):
    check_type(1, x, (ltypes.Number,))
    minarg = x
    for i, arg in enumerate(rest, 2):
        check_type(i, arg, (ltypes.Number,))
        if arg.f < minarg.f:
            minarg = arg
    return minarg
    yield

@register_module_function("math")
@require_params(0)
def lua_math__mininteger(_):
    return ltypes.Number(float(-sys.maxsize-1))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_modf(_, x):
    check_type(1, x, (ltypes.Number,))
    frac, inte = math.modf(x.f)
    return ltypes.Number(float(inte)), ltypes.Number(float(frac))
    yield

@register_module_function("math")
@require_params(0)
def lua_math__pi(_):
    return ltypes.Number(math.pi)
    yield

@register_module_function("math")
@require_params(1)
def lua_math_rad(_, x):
    check_type(1, x, (ltypes.Number,))
    return ltypes.Number(math.radians(x.f))
    yield

@register_module_function("math")
@require_params(0, 2)
def lua_math_random(_, m=None, n=None):
    if m is None:
        return ltypes.Number(random.random())
    check_type(1, m, (ltypes.Number,))
    if n is None:
        return ltypes.Number(float(random.randint(1, int(m.f))))
    check_type(2, n, (ltypes.Number,))
    if int(n.f) < int(m.f):
        raise ltypes.InterpreterError("random second argument must be greater or equal to first")
    return ltypes.Number(float(random.randint(int(m.f), int(n.f))))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_seed(_, x):
    check_type(1, x, (ltypes.Number,))
    random.seed(int(x.f))
    return ()
    yield

@register_module_function("math")
@require_params(1)
def lua_math_sin(_, x):
    return ltypes.Number(math.sin(x.f))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_sqrt(_, x):
    check_type(1, x, (ltypes.Number,))
    try:
        return ltypes.Number(math.sqrt(x.f))
    except ValueError:
        return ltypes.Number(float("nan"))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_tan(_, x):
    check_type(1, x, (ltypes.Number,))
    try:
        return ltypes.Number(math.tan(x.f))
    except ValueError:
        return ltypes.Number(float("nan"))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_tointeger(_, x):
    check_type(1, x, (ltypes.Number,))
    return ltypes.Number(float(int(x.f)))
    yield

@register_module_function("math")
@require_params(1)
def lua_math_type(_, x):
    return (ltypes.String(b"integer") if x.f.is_integer() else ltypes.String(b"float")) if isinstance(x, ltypes.Number) else ltypes.nil
    yield

@register_module_function("math")
@require_params(2)
def lua_math_ult(_, m, n):
    check_type(1, m, (ltypes.Number,))
    check_type(2, n, (ltypes.Number,))
    m = int(m.f) % sys.maxsize
    n = int(n.f) % sys.maxsize
    return ltypes.true if m < n else ltypes.false
    yield
