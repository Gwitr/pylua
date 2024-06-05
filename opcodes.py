# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

import collections
from typing import ClassVar
from dataclasses import dataclass, field

@dataclass(repr=False)
class Opcode:
    arg: "int | str | Label | ClosureInfo | None"
    string_arg: ClassVar[bool]
    op: ClassVar[str]

    def __repr__(self):
        if self.__class__.arg is not None:
            return f"{self.__class__.__name__}"
        return f"{self.__class__.__name__} {self.arg!r}"

    def __post_init__(self):
        if self.arg is None:
            raise ValueError("opcode arg must not be none")

class Label:

    def __sub__(self, other):
        if not isinstance(other, Label):
            return NotImplemented
        return LabelDifference(self, other)

@dataclass
class LabelDifference:
    lhs: Label
    rhs: Label
    offs: int = 0

    def __add__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return LabelDifference(self.lhs, self.rhs, self.offs + other)

    def __sub__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return LabelDifference(self.lhs, self.rhs, self.offs - other)

@dataclass
class LabelTarget:
    label: Label

class StringRef(collections.UserString):
    __hash__ = None

    def set(self, data):
        self.data = data

@dataclass
class ClosureInfo:
    localvars: set = field(default_factory=set)

def opcode_factory(cls_name, predef_op, predef_arg=None, *, is_string_arg=False):
    @dataclass(repr=False)
    class Subclass(Opcode):
        op = predef_op
        string_arg = is_string_arg
        arg: int | str | None = predef_arg
    Subclass.__name__ = Subclass.__qualname__ = cls_name
    return Subclass

def encode(code):
    strings = []
    def add_string(s):
        try:
            return strings.index(s)
        except ValueError:
            if len(strings) == 9999:
                raise ValueError("chunk has over 9999 strings") from None
            strings.append(s)
            return len(strings) - 1

    labels = {}
    ip = 0
    for op in code:
        if isinstance(op, LabelTarget):
            labels[op.label] = ip
        else:
            ip += 1

    result = ""
    for op in code:
        if isinstance(op, LabelTarget):
            continue
        arg = op.arg
        if isinstance(arg, str):
            arg = add_string(arg)

        elif isinstance(arg, ClosureInfo):
            arg = add_string("".join(str(add_string(i)).zfill(4) for i in arg.localvars))

        elif isinstance(arg, Label):
            arg = labels[arg]

        elif isinstance(arg, LabelDifference):
            arg = labels[arg.lhs] - labels[arg.rhs]

        if not 0 <= arg < 9999:
            raise ValueError(f"{op} arg(={arg}) out of range")
        result += op.op + str(arg).zfill(4)
    return "$" + "".join(f"{len(string):04d}{string}" for string in strings) + ";" + result

def decode(bytecode):
    strings = []
    i = 1
    while bytecode[i] != ";":
        l = int(bytecode[i:i+4].lstrip("0") or "0")
        i += 4
        strings.append(bytecode[i:i+l])
        i += l

    type_map = {(cls.op, getattr(cls, "arg", None)): cls for cls in Opcode.__subclasses__()}
    predef_arg = {cls.op for cls in Opcode.__subclasses__() if cls.arg is not None}

    ops = []
    j = 0
    i += 1
    while i < len(bytecode):
        op, *rest = bytecode[i:i+5]
        arg = int("".join(rest).lstrip("0") or "0")
        cls = type_map[op, (arg if op in predef_arg else None)]
        if cls.string_arg:
            arg = strings[arg]
        ops.append((cls(arg), j))
        i += 5
        j += 5
    return strings, ops

Nop = opcode_factory("Nop", " ", 0)

SetDual = opcode_factory("SetDual", ":")

RefName = opcode_factory("RefName", "n", is_string_arg=True)
RefItem = opcode_factory("RefItem", "i", 0)
GetName = opcode_factory("GetName", "z", is_string_arg=True)
GetItem = opcode_factory("GetItem", "y", 0)

PushString = opcode_factory("PushString", "c", is_string_arg=True)
PushNumber = opcode_factory("PushNumber", "N", is_string_arg=True)
PushFunction = opcode_factory("PushFunction", "f")
PushInline = opcode_factory("PushInline", "'")  # arg lists number of bytecode characters to skip and push as a string
PushNil = opcode_factory("PushNil", "s", 0)
PushTrue = opcode_factory("PushTrue", "s", 1)
PushFalse = opcode_factory("PushFalse", "s", 2)
# TODO: Consider giving this a fmt argument, to limit the amount of stack access
# necessary ("sspspspps*" would mean sequence-item, sequence-item, pair-item, ..., pair-item, sequence-item, bounded-sequence)
PushTable = opcode_factory("PushTable", "t")
PushVarargs = opcode_factory("PushVarargs", "*", 0)

Swap = opcode_factory("Swap", "S", 0)
Dup = opcode_factory("Dup", "2")

PushBound = opcode_factory("PushBound", "b", 0)
PopBound = opcode_factory("PopBound", "B", 0)
DiscardBound = opcode_factory("DiscardBound", "~", 0)
SetTop = opcode_factory("SetTop", "#")

Call = opcode_factory("Call", "$")
Return = opcode_factory("Return", "!")

Add = opcode_factory("Add", "o", 0)
Subtract = opcode_factory("Subtract", "o", 1)
Multiply = opcode_factory("Multiply", "o", 2)
Divide = opcode_factory("Divide", "o", 3)
Negative = opcode_factory("Negative", "o", 4)
Len = opcode_factory("Len", "o", 5)
Equal = opcode_factory("Equal", "o", 6)
Less = opcode_factory("Less", "o", 7)
Greater = opcode_factory("Greater", "o", 8)
LessEqual = opcode_factory("LessEqual", "o", 9)
GreaterEqual = opcode_factory("GreaterEqual", "o", 10)
Unequal = opcode_factory("Unequal", "o", 11)
Sign = opcode_factory("Sign", "o", 12)
# TODO: Proper short-circuiting OR/AND behavior
Or = opcode_factory("Or", "o", 13)
And = opcode_factory("And", "o", 14)
Not = opcode_factory("Not", "o", 15)
Concat = opcode_factory("Concat", "o", 16)

PushBlock = opcode_factory("PushBlock", "+")
PopBlock = opcode_factory("PopBlock", "-", 0)
BlockInfo = opcode_factory("BlockInfo", "I", is_string_arg=True)
UpvalueInfo = opcode_factory("UpvalueInfo", "^", is_string_arg=True)
Break = opcode_factory("Break", "x", 0)

If = opcode_factory("If", "?", 0)
IfElse = opcode_factory("IfElse", "?", 1)
Jump = opcode_factory("Jump", ">")
