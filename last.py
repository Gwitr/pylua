# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring,unnecessary-lambda-assignment,unnecessary-lambda,line-too-long

from typing import ClassVar

import opcodes
from opcodes import Label, LabelTarget, ClosureInfo

LocalDecl = opcodes.opcode_factory("LocalDecl", "\0", 0)  # Fake opcode that only exists during parsing

class Node:
    code: list[opcodes.Opcode]
    DEFAULT_TOP: ClassVar[int | None] = 1

    def copy(self, **kwargs):
        node = Node.__new__(self.__class__)
        for k, v in self.__dict__.items():
            setattr(node, k, kwargs.get(k, v))
        return node

    def with_ctx(self, *, top=None, **kwargs):
        if kwargs:
            raise ValueError(kwargs)
        res = self
        if top is not None and top != self.DEFAULT_TOP:
            res = res.copy(code=[opcodes.PushBound(), *self.code, opcodes.SetTop(top), opcodes.DiscardBound()])
        return res

    def __repr__(self):
        return f"{self.__class__.__name__}({self.code})"

class Statement(Node):
    DEFAULT_TOP = 0

class Locals(Statement):

    def __init__(self, variables: list[str]):
        self.code = [LocalDecl(var) for var in variables]

class LocalAssign(Statement):

    def __init__(self, variables: list[str], what: Node):
        self.code = [
            *(LocalDecl(var) for var in variables),
            *(opcodes.RefName(var) for var in variables),
            *what.with_ctx(top=len(variables)).code,
            opcodes.SetDual(len(variables))
        ]

class Assign(Statement):

    def __init__(self, variables: list[Node], what: Node):
        self.code = [
            *(opcode for var in variables for opcode in var.with_ctx(lvalue=True).code),
            *what.with_ctx(top=len(variables)).code,
            opcodes.SetDual(len(variables))
        ]

class For(Statement):

    def __init__(self, name: str, start_value: Node, end_value: Node, step: Node | None, block: Node):
        # TODO: Iterator for
        self.code = [
            opcodes.PushBlock(label := Label()), opcodes.BlockInfo(ClosureInfo([name])),

            opcodes.PushBound(), opcodes.RefName(name), *start_value.with_ctx(top=1).code, opcodes.SetDual(1),

            *end_value.with_ctx(top=1).code,
            *(step.with_ctx(top=1).code if step else [opcodes.PushNumber("1")]),

            LabelTarget(jump_target := Label()), opcodes.PushBound(),
                opcodes.Dup(1), opcodes.Sign(),
                opcodes.Dup(0), opcodes.Dup(4), opcodes.Multiply(),
                opcodes.GetName(name), opcodes.Dup(2), opcodes.Multiply(),
                opcodes.GreaterEqual(), opcodes.Swap(),
            opcodes.SetTop(1), opcodes.DiscardBound(),

            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start),
            *block.code,
            opcodes.RefName(name), opcodes.GetName(name), opcodes.Dup(2), opcodes.Add(), opcodes.SetDual(1),
            opcodes.Jump(jump_target),

            LabelTarget(end), opcodes.If(),
            opcodes.PopBlock(),
            LabelTarget(label), opcodes.PopBound()
        ]

class While(Statement):

    def __init__(self, cond: Node, body: Node):
        self.code = [
            LabelTarget(target := Label()), *cond.with_ctx(top=1).code,
            opcodes.PushInline((end := Label()) - (start := Label())),
            LabelTarget(start), *body.with_ctx(allow_break=True).code,
            opcodes.Jump(target),
            LabelTarget(end),
            opcodes.If()
        ]

# TODO: elseif
class If(Statement):

    def __init__(self, cond: Node, if_: Node):
        self.code = [
            *cond.with_ctx(top=1).code,
            opcodes.PushInline((end := Label()) - (start := Label())),
            LabelTarget(start), *if_.code,
            opcodes.Jump(jump1 := Label()),
            LabelTarget(end), opcodes.If(),
            LabelTarget(jump1),
        ]

class IfElse(Statement):

    def __init__(self, cond: Node, if_: Node, else_: Node):
        self.code = [
            *cond.with_ctx(top=1).code,
            opcodes.PushInline((end := Label()) - (start := Label())),
            LabelTarget(start), *if_.code,
            opcodes.Jump(jump1 := Label()),
            LabelTarget(end), opcodes.PushInline((end2 := Label()) - (start2 := Label())),
            LabelTarget(start2), *else_.code,
            opcodes.Jump(jump2 := Label()),
            LabelTarget(end2), opcodes.IfElse(),
            LabelTarget(jump1), LabelTarget(jump2)
        ]

class Return(Statement):

    def __init__(self, exprs: list[Node]):
        self.code = [
            opcodes.PushBound(),
            *(opcode for expr in exprs[:-1] for opcode in expr.with_ctx(top=1).code),
            *(opcode for expr in exprs[-1:] for opcode in expr.code),
            opcodes.Return(0)
        ]

class Break(Statement):

    def __init__(self):
        self.code = [opcodes.Break()]

class Block(Statement):
    localvars: list[str]
    block_code: list[opcodes.Opcode]

    def __init__(self, stmts: list[Node]):
        block_code = [opcode for stmt in stmts for opcode in stmt.with_ctx(top=0).code]
        localvars = [i.arg for i in block_code if isinstance(i, LocalDecl)]
        block_code = [i for i in block_code if not isinstance(i, LocalDecl)]
        self.localvars = localvars
        self.block_code = block_code
        self.code = [
            opcodes.PushBlock(0),
            opcodes.BlockInfo(ClosureInfo(localvars)),
            *(opcode for stmt in stmts for opcode in stmt.with_ctx(top=0).code),
            opcodes.PopBlock()
        ]

    def with_ctx(self, allow_break=False, function=False, **kwargs):
        assert not (function and allow_break)
        res = self
        if allow_break:
            res = res.copy(code=[
                opcodes.PushBlock(label := Label()),
                opcodes.BlockInfo(ClosureInfo(self.localvars)),
                *self.block_code,
                opcodes.PopBlock(),
                LabelTarget(label), opcodes.Nop()  # TODO: Get rid of this Nop somehow
            ])
        if function:
            res = res.copy(code=[
                opcodes.PushBlock(0),
                opcodes.BlockInfo(ClosureInfo(self.localvars)),
                *self.block_code,
                opcodes.PushBound(),
                opcodes.Return(0)
            ])
        return super(__class__, res).with_ctx(**kwargs)


class FunctionBase(Statement):
    DEFAULT_TOP = 0

    def __init__(self, args: list[str], body: Node):
        self.code = [
            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start), *body.with_ctx(function=True).code,
            LabelTarget(end), *(opcodes.PushString(arg) for arg in args),
            opcodes.PushFunction(len(args))
        ]

class FunctionDef(FunctionBase):

    def __init__(self, target: Node, args: list[str], body: Node):
        super().__init__(args, body)
        self.code = [*target.with_ctx(lvalue=True).code, *self.code, opcodes.SetDual(1)]

class LocalFunctionDef(FunctionBase):

    def __init__(self, target: str, args: list[str], body: Node):
        super().__init__(args, body)
        self.code = [LocalDecl(target), opcodes.RefName(target), *self.code, opcodes.SetDual(1)]

class SelfFunctionDef(FunctionBase):

    def __init__(self, target: Node, args: list[str], body: Node):
        super().__init__(["self"] + args, body)
        self.code = [*target.with_ctx(lvalue=True).code, *self.code, opcodes.SetDual(1)]

class InlineFunction(FunctionBase):
    DEFAULT_TOP = 1

class Call(Node):
    DEFAULT_TOP = None

    def __init__(self, what: Node, args: list[Node]):
        self.code = [
            *what.with_ctx(top=1).code,
            opcodes.PushBound(),
            *(opcode for arg in args[:-1] for opcode in arg.with_ctx(top=1).code),
            *(opcode for arg in args[-1:] for opcode in arg.code),
            opcodes.Call(0)
        ]

class Data(Node):
    DATA_OP: ClassVar[type[opcodes.Opcode]]

    def __init__(self, data: str | int | None = None):
        self.code = [self.DATA_OP()] if data is None else [self.DATA_OP(data)]

class Ref(Node):
    REF_OP: ClassVar[type[opcodes.Opcode]]

    def with_ctx(self, *, lvalue=False, **kwargs):
        return self.copy(code=[*self.code[:-1], self.REF_OP(self.code[-1].arg)]) if lvalue else self

class Index(Ref):
    REF_OP = opcodes.RefItem

    def __init__(self, what: Node, idx: Node):
        self.code = [*what.with_ctx(top=1).code, *idx.with_ctx(top=1).code, opcodes.GetItem()]

class NameRef(Ref, Data):
    REF_OP = opcodes.RefName
    DATA_OP = opcodes.GetName

class BinaryOperator(Node):
    OPERATOR_OPCODE: ClassVar[type[opcodes.Opcode]]
    lhs: Node
    rhs: Node

    def __init__(self, lhs: Node, rhs: Node):
        # We can't simply discard these, as they're used in the right-recursion correction thingy
        self.lhs = lhs
        self.rhs = rhs
        self.code = [*lhs.with_ctx(top=1).code, *rhs.with_ctx(top=1).code, self.OPERATOR_OPCODE()]

class Add(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Add

class Subtract(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Subtract

class Multiply(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Multiply

class Divide(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Divide

class Concat(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Concat

class Less(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Less

class Greater(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Greater

class LessEqual(BinaryOperator):
    OPERATOR_OPCODE = opcodes.LessEqual

class GreaterEqual(BinaryOperator):
    OPERATOR_OPCODE = opcodes.LessEqual

class Equal(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Equal

class NotEqual(BinaryOperator):
    OPERATOR_OPCODE = opcodes.Unequal

class Parenthesized(Node):

    def __init__(self, expr: Node):
        self.code = expr.with_ctx(top=1).code

class Length(Node):

    def __init__(self, what: Node):
        self.code = [*what.with_ctx(top=1).code, opcodes.Len()]

class Negative(Node):

    def __init__(self, what: Node):
        self.code = [*what.with_ctx(top=1).code, opcodes.Negative()]

class Table(Node):

    def __init__(self, data: tuple[tuple[Node, Node] | Node]):
        if data and not isinstance(data[-1], tuple):
            self.code = [
                *(opcode for elem in data[:-1] for opcode in ([*elem[0].with_ctx(top=1).code, *elem[1].with_ctx(top=1).code] if isinstance(elem, tuple) else [opcodes.PushNil(), *elem.with_ctx(top=1).code])),
                opcodes.PushBound(),
                *data[-1].code,
                opcodes.PushTable(len(data) - 1)
            ]
        else:
            self.code = [
                *(opcode for elem in data for opcode in ([*elem[0].with_ctx(top=1).code, *elem[1].with_ctx(top=1).code] if isinstance(elem, tuple) else [opcodes.PushNil(), *elem.with_ctx(top=1).code])),
                opcodes.PushBound(),
                opcodes.PushTable(len(data))
            ]

class Number(Data):
    DATA_OP = opcodes.PushNumber

class String(Data):
    DATA_OP = opcodes.PushString

class Vararg(Data):
    DEFAULT_TOP = None
    DATA_OP = opcodes.PushVarargs

class SpecialNil(Data):
    DATA_OP = opcodes.PushNil

class SpecialTrue(Data):
    DATA_OP = opcodes.PushTrue

class SpecialFalse(Data):
    DATA_OP = opcodes.PushFalse

class Chunk(Node):

    def __init__(self, body: Node):
        self.code = [
            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start), *body.with_ctx(function=True).code,
            LabelTarget(end), opcodes.PushFunction(0)
        ]
