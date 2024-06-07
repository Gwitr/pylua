"""Defines all Lua types"""

import abc
import inspect
import functools
from typing import Any, ClassVar
from dataclasses import dataclass

import opcodes

class InterpreterError(Exception):
    """Any error raised by the interpreter"""

class LuaType(abc.ABC):
    """Base class for all Lua types"""
    TYPE: ClassVar[str]  # for interpreter introspection

    @property
    @abc.abstractmethod
    def type(self):
        """The type of the value as a Lua string"""
        # for Lua introspection

    @abc.abstractmethod
    def tostring(self):
        """Converts this into a String instance"""

class _Nil(LuaType):
    """The class for the representation of the Lua nil singleton"""
    TYPE = "nil"

    @property
    def type(self):
        return String(b"nil")

    def tostring(self):
        return String(b"nil")

    def __repr__(self):
        return "nil"

@dataclass(frozen=True, slots=True, eq=True)
class Boolean(LuaType):
    """Represents a Lua boolean"""
    TYPE = "boolean"
    b: bool

    @property
    def type(self):
        return String(b"boolean")

    def tostring(self):
        return String(b"true" if self.b else b"false")

nil = _Nil()
true = Boolean(True)
false = Boolean(False)

@dataclass(frozen=True, slots=True, eq=True)
class Number(LuaType):
    """Represents a Lua number"""
    TYPE = "number"
    f: float

    @property
    def type(self):
        return String(b"number")

    def tostring(self):
        return String(str(self.f).encode("ascii"))

@dataclass(frozen=True, slots=True, eq=True)
class String(LuaType):
    """Represents a Lua string"""
    TYPE = "string"
    s: bytes

    @property
    def type(self):
        return String(b"string")

    def tostring(self):
        return self

    def getn(self):
        """length of string"""
        return Number(float(len(self.s)))

    def unicode(self):
        """debug only"""
        return self.s.decode('ascii', errors='backslashreplace')

    def __post_init__(self):
        assert isinstance(self.s, bytes)

class Table(LuaType):
    """Represents a Lua table"""
    TYPE = "table"

    @property
    def type(self):
        return String(b"table")

    def tostring(self):
        return String(f"table: 0x{id(self):08x}".encode("ascii"))

    def __init__(self, from_=None):
        self.seq = []
        self.map = {}
        self._metatable = nil
        if from_ is not None:
            self.update(from_)

    def __setitem__(self, key, value):
        assert isinstance(key, LuaType)
        assert isinstance(value, LuaType)
        if key is nil:
            raise InterpreterError("assignment to nil table key")
        if isinstance(key, Number) and key.f.is_integer() and key.f > 0:
            loc = int(key.f) - 1
            if loc >= len(self.seq):
                if value is not nil:
                    for _ in range(loc - len(self.seq)):
                        self.seq.append(nil)
                    self.seq.append(value)
            else:
                self.seq[loc] = value
            while self.seq and self.seq[-1] is nil:
                del self.seq[-1]
        else:
            self.map[key] = value
            keys = list(self.map.keys())
            # TODO: To not break next(...), we only ever delete elements if they're at the end of the keys. This is not very memory-efficient.
            while keys and self.map[keys[-1]] is nil:
                del self.map[keys[-1]]
                del keys[-1]

    def __getitem__(self, key):
        assert isinstance(key, LuaType)
        if isinstance(key, Number) and key.f.is_integer() and key.f > 0:
            loc = int(key.f) - 1
            if loc >= len(self.seq):
                return nil
            return self.seq[loc]
        return self.map.get(key, nil)

    REPR_CACHE = None
    def __repr__(self):
        if Table.REPR_CACHE is None:
            cleanup = True
            Table.REPR_CACHE = set()
        else:
            cleanup = False
        try:
            Table.REPR_CACHE.add(id(self))
            seqrepr = ["..." if id(i) in Table.REPR_CACHE else repr(i) for i in self.seq]
            maprepr = [(f"{k.unicode()}=..." if id(v) in Table.REPR_CACHE else f"{k.unicode()}={v!r}") if isinstance(k, String) else (f"[{k!r}]=..." if id(v) in Table.REPR_CACHE else f"[{k!r}]={v!r}") for k, v in self.map.items()]
            return "{" + ", ".join(seqrepr + maprepr) + "}"
        finally:
            if cleanup:
                Table.REPR_CACHE = None

    def nextkey(self, key=nil):
        """Acts like Lua's next(...) function but only for keys"""
        if key is nil:
            if self.seq:
                return Number(1.0)
            if self.map:
                return list(self.map.keys())[0]
            return nil
        if isinstance(key, Number) and key.f.is_integer() and key.f > 0:
            if int(key.f) == len(self.seq):
                if self.map:
                    return list(self.map.keys())[0]
                return nil
            return Number(key.f + 1.0)
        # TODO: This is really inefficient
        keys = list(self.map.keys())
        try:
            return keys[keys.index(key) + 1]
        except (ValueError, IndexError):
            return nil

    @property
    def metatable(self):
        """The metatable of this table"""
        return self._metatable

    @metatable.setter
    def metatable(self, _mt):
        raise NotImplementedError

    def getn(self):
        """Gets a boundary of this table"""
        return Number(float(len(self.seq)))

    def update(self, other):
        for k, v in dict(other).items():
            assert isinstance(k, LuaType)
            assert isinstance(v, LuaType)
            self[k] = v

@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Function(LuaType):
    """Represents a Lua function"""
    TYPE = "function"
    interpreter: Any
    argnames: tuple[bytes]
    offs: int
    bytecode: tuple[list[bytes], list[opcodes.Opcode]]
    upvalues: dict[str, Table]

    @property
    def type(self):
        return String(b"function")

    def tostring(self):
        return String(f"function: 0x{id(self):08x}".encode("ascii"))

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"func({', '.join(i.decode('ascii', errors='backslashreplace') for i in self.argnames)})"

@dataclass(slots=True, eq=True, frozen=True, repr=False)
class ThreadContainer(LuaType):
    """Represents a Lua thread"""
    TYPE = "thread"
    thread: Any   # linterp.Thread; not referenced directly to avoid circular import

    @property
    def type(self):
        return String(b"thread")

    def tostring(self):
        return String(f"thread: 0x{id(self.thread):08x}".encode("ascii"))

    def __repr__(self):
        return "thread"

## Types not possible to instantiate in Lua code

class BuiltinFunction(LuaType):
    """Represents a Python function exposed to Lua"""
    TYPE = "function"

    @property
    def type(self):
        return String(b"function")

    def tostring(self):
        return String(f"function: 0x{id(self.f):08x}".encode("ascii"))

    def __init__(self, interpreter, f):
        self.f = functools.partial(f, interpreter)
        try:
            self.name = f.__qualname__
        except AttributeError:
            self.name = "?"

    def __eq__(self, other):
        if not isinstance(other, BuiltinFunction):
            return NotImplemented
        return self.f == other.f

    def __repr__(self):
        return f"func(...): {self.name}"

@dataclass(frozen=True,slots=True,eq=True)
class Code(LuaType):
    """Represents an offset into bytecode"""
    TYPE = "code"
    offs: int

    @property
    def type(self):
        raise InterpreterError("interrogate type of bytecode offset")

    def tostring(self):
        raise InterpreterError("tostring(...) invoked on bytecode offset")

@dataclass(frozen=True, slots=True, eq=True, repr=False)
class Ref(LuaType):
    """Represents a reference to a dictionary or table item (used for assignments)"""
    TYPE = "ref"
    obj: dict | Table
    idx: int | LuaType

    @property
    def type(self):
        raise InterpreterError("interrogate type of ref")

    def tostring(self):
        raise InterpreterError("tostring(...) invoked on ref")

    def set(self, value):
        """Assign to reference"""
        assert isinstance(value, LuaType)
        self.obj[self.idx] = value

    def __repr__(self):
        return f"...[{self.idx}]"
