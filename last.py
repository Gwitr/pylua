# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring,unnecessary-lambda-assignment,unnecessary-lambda,line-too-long

# TODO: Dynamically compile the grammar down to Python (There's huge overhead here somewhere and I can't find it no matter how hard I look)

from typing import Any, ClassVar
from dataclasses import dataclass

import opcodes
from llexer import ParseError, Tokens
from opcodes import Label, LabelTarget, ClosureInfo

# import sys; sys.setrecursionlimit(100)
DEBUG = True

## Opcode stuff
LocalDecl = opcodes.opcode_factory("LocalDecl", "\0", 0)  # Fake opcode that only exists during parsing

## AST boilerplate
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

class Locals(Node):
    DEFAULT_TOP = 0

    def __init__(self, variables: list[str]):
        self.code = [LocalDecl(var) for var in variables]

class LocalAssign(Node):
    DEFAULT_TOP = 0

    def __init__(self, variables: list[str], what: Node):
        self.code = [
            *(LocalDecl(var) for var in variables),
            *(opcodes.RefName(var) for var in variables),
            *what.with_ctx(top=len(variables)).code,
            opcodes.SetDual(len(variables))
        ]

class Assign(Node):
    DEFAULT_TOP = 0

    def __init__(self, variables: list[Node], what: Node):
        self.code = [
            *(opcode for var in variables for opcode in var.with_ctx(lvalue=True).code),
            *what.with_ctx(top=len(variables)).code,
            opcodes.SetDual(len(variables))
        ]

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

class Index(Node):

    def __init__(self, what: Node, idx: Node):
        self.code = [*what.with_ctx(top=1).code, *idx.with_ctx(top=1).code, opcodes.GetItem()]

    def with_ctx(self, *, lvalue=False, **kwargs):
        res = self
        if lvalue:
            res = res.copy(code=[*res.code[:-1], opcodes.RefItem()])
        return super(__class__, res).with_ctx(**kwargs)

class NameRef(Node):

    def __init__(self, name: str):
        self.code = [opcodes.GetName(name)]

    def with_ctx(self, *, lvalue=False, **kwargs):
        res = self
        if lvalue:
            res = res.copy(code=[opcodes.RefName(self.code[-1].arg)])
        return super(__class__, res).with_ctx(**kwargs)

class BinaryOperator(Node):
    OPERATOR_OPCODE: ClassVar[type[opcodes.Opcode]]

    lhs: Node
    rhs: Node
    def __init__(self, lhs: Node, rhs: Node):
        # We can't simply discard these, as they're used in later code
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

class Number(Node):

    def __init__(self, data: str):
        self.code = [opcodes.PushNumber(data)]

class String(Node):

    def __init__(self, data: str):
        self.code = [opcodes.PushString(data)]

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

class Vararg(Node):
    DEFAULT_TOP = None

    def __init__(self):
        self.code = [opcodes.PushVarargs()]

class SpecialNil(Node):

    def __init__(self):
        self.code = [opcodes.PushNil()]

class SpecialTrue(Node):

    def __init__(self):
        self.code = [opcodes.PushTrue()]

class SpecialFalse(Node):

    def __init__(self):
        self.code = [opcodes.PushFalse()]

class FunctionDef(Node):
    DEFAULT_TOP = 0

    def __init__(self, target: Node, args: list[str], body: Node):
        self.code = [
            *target.with_ctx(lvalue=True).code,
            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start), *body.with_ctx(function=True).code,
            LabelTarget(end), *(opcodes.PushString(arg) for arg in args),
            opcodes.PushFunction(len(args)),
            opcodes.SetDual(1)
        ]

class LocalFunctionDef(Node):
    DEFAULT_TOP = 0

    def __init__(self, target: str, args: list[str], body: Node):
        self.code = [
            LocalDecl(target),
            opcodes.RefName(target),
            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start), *body.with_ctx(function=True).code,
            LabelTarget(end), *(opcodes.PushString(arg) for arg in args),
            opcodes.PushFunction(len(args)),
            opcodes.SetDual(1)
        ]

class SelfFunctionDef(Node):
    DEFAULT_TOP = 0

    def __init__(self, target: Node, args: list[str], body: Node):
        self.code = [
            *target.with_ctx(lvalue=True).code,
            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start), *body.with_ctx(function=True).code,
            LabelTarget(end), *(opcodes.PushString(arg) for arg in ["self", *args]),
            opcodes.PushFunction(len(args)),
            opcodes.SetDual(1)
        ]

class InlineFunction(Node):
    DEFAULT_TOP = 1

    def __init__(self, args: list[str], body: Node):
        self.code = [
            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start), *body.with_ctx(function=True).code,
            LabelTarget(end), *(opcodes.PushString(arg) for arg in args),
            opcodes.PushFunction(len(args))
        ]

class For(Node):
    DEFAULT_TOP = 0

    def __init__(self, name: str, start_value: Node, end_value: Node, step: Node | None, block: Node):
        # TODO: Iterator for
        self.code = [
            opcodes.PushBlock(label := Label()),
            opcodes.BlockInfo(ClosureInfo([name])),

            opcodes.PushBound(),
            opcodes.RefName(name),
            *start_value.with_ctx(top=1).code,
            opcodes.SetDual(1),

            *end_value.with_ctx(top=1).code,
            *(step.with_ctx(top=1).code if step else [opcodes.PushNumber("1")]),

            LabelTarget(jump_target := Label()), opcodes.PushBound(),
            opcodes.Dup(1),
            opcodes.Sign(),
            opcodes.Dup(0),
            opcodes.Dup(4),
            opcodes.Multiply(),
            opcodes.GetName(name),
            opcodes.Dup(2),
            opcodes.Multiply(),
            opcodes.GreaterEqual(),
            opcodes.Swap(),
            opcodes.SetTop(1),
            opcodes.DiscardBound(),

            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start),
            *block.code,
            opcodes.RefName(name),
            opcodes.GetName(name),
            opcodes.Dup(2),
            opcodes.Add(),
            opcodes.SetDual(1),
            opcodes.Jump(jump_target),

            LabelTarget(end),

            opcodes.If(),
            opcodes.PopBlock(),
            LabelTarget(label), opcodes.PopBound()
        ]

class While(Node):
    DEFAULT_TOP = 0

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
class If(Node):
    DEFAULT_TOP = 0

    def __init__(self, cond: Node, if_: Node):
        self.code = [
            *cond.with_ctx(top=1).code,
            opcodes.PushInline((end := Label()) - (start := Label())),
            LabelTarget(start), *if_.code,
            opcodes.Jump(jump1 := Label()),
            LabelTarget(end), opcodes.If(),
            LabelTarget(jump1),
        ]

class IfElse(Node):
    DEFAULT_TOP = 0

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

class Return(Node):
    DEFAULT_TOP = 0

    def __init__(self, exprs: list[Node]):
        self.code = [
            opcodes.PushBound(),
            *(opcode for expr in exprs[:-1] for opcode in expr.with_ctx(top=1).code),
            *(opcode for expr in exprs[-1:] for opcode in expr.code),
            opcodes.Return(0)
        ]

class Break(Node):
    DEFAULT_TOP = 0

    def __init__(self):
        self.code = [opcodes.Break()]

class Block(Node):
    DEFAULT_TOP = 0

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

class Chunk(Node):

    def __init__(self, body: Node):
        self.code = [
            opcodes.PushInline((end := Label()) - (start := Label()) - 1),
            LabelTarget(start), *body.with_ctx(function=True).code,
            LabelTarget(end), opcodes.PushFunction(0)
        ]

## Parser boilerplate
@dataclass(frozen=True)
class ParserState:
    """Monad representing the state of a parser"""

@dataclass(frozen=True)
class OKParserState(ParserState):
    """ParserState subclass representing a state of an active parser"""
    tokens: Tokens
    results: tuple[Any, ...]

@dataclass(frozen=True)
class ErrorParserState(ParserState):
    """ParserState subclass representing a state of a parser that encountered an error"""
    toks: Tokens
    pos: tuple[str, int, int]
    expected: frozenset[str]
    op_stack: tuple[str] = ()

UnknownLeadingToken = object()

# 11629/154    0.010    0.000    0.066    0.000
# 12349/154    0.014    0.000    0.064    0.000
#  4698/154    0.007    0.000    0.053    0.000
#  10660/83    0.010    0.000    0.068    0.001
#   2923/83    0.004    0.000    0.029    0.000
#   2871/85    0.004    0.000    0.026    0.000
#   2600/85    0.003    0.000    0.026    0.000

class _Pipeline:
    """Helper class for constructing parser pipelines
    (helps to cut down on the amount of `(lambda tok, rest: OKParserState(tok, rest) >> ...)` expressions)
    (and makes it harder to hit the recursion limit)"""

    PIPELINES = set()

    def __repr__(self):
        return f">>({self.name})"

    def __init__(self, fs, name="?", expect=UnknownLeadingToken):
        self.PIPELINES.add(self)
        self.fs = fs
        self.name = name
        self.expect = expect

    def __rshift__(self, f):
        if self.fs is None:
            expect = (frozenset(f.expect) if isinstance(f.expect, frozenset) else None) if isinstance(f, _Pipeline) else UnknownLeadingToken
            if isinstance(f, _Pipeline):
                return _Pipeline(f.fs[:], self.name, expect)
            return _Pipeline([f], self.name, expect)
        if isinstance(f, _Pipeline):
            return _Pipeline([*self.fs, *f.fs], self.name, self.expect)
        return _Pipeline([*self.fs, f], self.name, self.expect)

    def __call__(self, toks, results=None):
        if results is None:
            return _Pipeline(self.fs, toks)
        state = OKParserState(toks, results)
        for f in self.fs:
            state = f(state.tokens, state.results)
            if isinstance(state, ErrorParserState):
                break
        return state

    def _resolve_fwd(self, stack):
        stack.append(self)
        i = 0
        while i < len(self.fs):
            step = self.fs[i]
            if isinstance(step, fwd):
                child = globals()[step.name]
                if child not in stack:
                    child._resolve_fwd(stack)
                    self.fs = [*self.fs[:i], *child.fs, *self.fs[i+1:]]
                    i += len(child.fs)
            i += 1
        stack.pop()
        return self

    @classmethod
    def resolve_fwd(cls):
        for pipeline in cls.PIPELINES:
            if pipeline.fs is None:
                continue
            pipeline._resolve_fwd([])

Pipeline = _Pipeline(None)

def ambig(*funcs):
    if not funcs:
        raise ValueError("ambig(...) takes at least 1 argument")

    def wrap(toks, prevs):
        result = None
        errors = []
        for func in funcs:
            result = func(toks, prevs)
            if isinstance(result, OKParserState):
                return result
            errors.append(result)
        final = max(errors, key=lambda x: x.pos[1:])
        return ErrorParserState(final.toks, final.pos, frozenset(i for err in errors if err.pos == final.pos for i in err.expected))

    expected_tokens = set()
    for branch in funcs:
        if (not isinstance(branch, _Pipeline)) or branch.expect in {UnknownLeadingToken, None}:
            expected_tokens = None
            break
        expected_tokens |= branch.expect

    pipeline = _Pipeline([wrap], "|".join(getattr(f, "name", "?") for f in funcs), None if expected_tokens is None else frozenset(expected_tokens))
    return pipeline

def ambig_lookahead(*branches):
    all_valid = set()
    order = [{}]
    for branch in branches:
        if branch is OKParserState:
            order.append(branch)
            order.append({})
            all_valid = None
        elif branch.expect is UnknownLeadingToken:
            raise ValueError(f"unknown leading token for branch {branch.name}")
        elif branch.expect is None:
            order.append(branch)
            order.append({})
            all_valid = None
        else:
            for tok in branch.expect:
                if all_valid is not None:
                    all_valid |= {tok}
                order[-1].setdefault(tok, []).append(branch)
    all_valid = None if all_valid is None else frozenset(all_valid)
    if DEBUG and all_valid is None:
        print("1 or more lack lookahead in", order)

    def wrap(toks, prevs):
        next_pos, next_tok = toks.advance()[0]
        funcs = [j for i in order for j in (i.get(next_tok[0], ()) if isinstance(i, dict) else [i])]
        if not funcs:
            return ErrorParserState(toks, next_pos, all_valid)
        return ambig(*funcs)(toks, prevs)

    return _Pipeline([wrap], "=(" + "|".join(getattr(f, "name", "?") for f in branches) + ")", all_valid)

def next_token(expect=None):
    expect = None if expect is None else frozenset(expect)
    def wrap(toks, prevs):
        (pos, tok), toks = toks.advance()
        if expect is None or tok[0] in expect:
            return OKParserState(toks, (*prevs, tok))
        return ErrorParserState(toks, pos, frozenset(expect))
    return _Pipeline([wrap], "next_token", expect)

@dataclass
class fwd:
    name: str

    @property
    def expect(self):
        return None

    def __call__(self, *args):
        # Not all fwd objects will be resolved (some are recursive)
        return globals()[self.name](*args)

# Constructs a parser that accepts the `func` parser if possible, but otherwise simply forwards the previous state
def optional(func):
    def wrap(toks, prevs):
        result = func(toks, prevs)
        if isinstance(result, ErrorParserState):
            return OKParserState(toks, prevs)
        return result
    return _Pipeline([wrap], "optional", None)

def optional_lookahead(func):
    def wrap(toks, prevs):
        cur_tok = toks.toks[toks.idx][1][0] if toks.idx < len(toks.toks) else "EOF"
        if func.expect is not None and cur_tok not in func.expect:
            return OKParserState(toks, prevs)
        result = func(toks, prevs)
        if isinstance(result, ErrorParserState):
            return OKParserState(toks, prevs)
        return result
    return _Pipeline([wrap], "optional_lookahead", None)

# Constructs a parser that removes N prior results, and pushes a new result generated by `nodefunc`
set_result = lambda n, nodefunc: _Pipeline([lambda toks, prevs: OKParserState(toks, (*(prevs[:-n] if n > 0 else prevs), nodefunc(prevs[-n:])))], "set_result", None)

# Constructs a parser that consumes a token, but doesn't do anything with it and doesn't modify the results list
def consume(expect=None):
    def wrap(toks, prevs):
        (pos, tok), toks = toks.advance()
        if expect is None or tok[0] in expect:
            return OKParserState(toks, prevs)
        return ErrorParserState(toks, pos, expect)
    return _Pipeline([wrap], "consume", frozenset(expect))

# Constructs a parser that consumes a token, invokes the `func` parser, and consumes another token
parse_surrounded = lambda func, left, right: Pipeline(f"surrounded({func!r},{left!r},{right!r})") >> consume({left}) >> func >> consume({right})

# Returns a function that parses a comma-separated list of things parsed by `func`
def parse_comma_list(func):
    def wrap(toks, prevs):
        state = func(toks, prevs)
        if isinstance(state, ErrorParserState):
            return state
        state = OKParserState(state.tokens, state.results[:-1] + ((state.results[-1],),))
        while True:
            new_state = consume({"COMMA"})(state.tokens, state.results)
            if isinstance(new_state, ErrorParserState):
                return state
            state = func(new_state.tokens, new_state.results)
            if isinstance(state, ErrorParserState):
                return state
            state = OKParserState(state.tokens, state.results[:-2] + (
                (*state.results[-2], state.results[-1]),
            ))
    return _Pipeline([wrap], f"comma-list({func!r})", func.expect)

# NOTE: does not consume results stack!!!
def parse_list(func):
    def wrap(toks, prevs):
        state = func(toks, prevs)
        # print("Advance", func, state)
        if isinstance(state, ErrorParserState):
            return state
        while True:
            new_state = func(state.tokens, state.results)
            # print("Advance", func, new_state)
            if isinstance(new_state, ErrorParserState):
                return state
            state = new_state
    return _Pipeline([wrap], f"list({func!r})", func.expect)

# NOTE: does not consume results stack!!!
def parse_sep_list(func, sep):
    def wrap(toks, prevs):
        state = func(toks, prevs)
        if isinstance(state, ErrorParserState):
            return state
        while True:
            new_state = consume({sep})(state.tokens, state.results)
            if isinstance(new_state, ErrorParserState):
                return state
            state = func(new_state.tokens, new_state.results)
            if isinstance(state, ErrorParserState):
                return state
    return _Pipeline([wrap], f"sep-list({func!r})", func.expect)

### Start of parser code

## Single token parsers

parse_name   = Pipeline("name")   >> next_token({"NAME"})   >> set_result(1, lambda prevs: NameRef(prevs[0][1]))
parse_string = Pipeline("string") >> next_token({"QSTR"})   >> set_result(1, lambda prevs: String(prevs[0][1]))
parse_number = Pipeline("number") >> next_token({"NUMBER"}) >> set_result(1, lambda prevs: Number(prevs[0][1]))

## Expression/subexpression parsers

parse_paren_expr_list = Pipeline("paren_expr_list") >> parse_surrounded(parse_comma_list(fwd("parse_expr")), "LPAREN", "RPAREN")
parse_call_suffix = ambig_lookahead(
    Pipeline("call-suffix[0]") >> consume({"LPAREN"}) >> consume({"RPAREN"}) >> set_result(0, lambda _: ()),
    Pipeline("call-suffix[1]") >> parse_string >> set_result(1, lambda prevs: (prevs[0],)),
    parse_paren_expr_list
)
parse_many_call_suffix = Pipeline("call-suffix*") >> parse_call_suffix >> set_result(2, lambda prevs: Call(*prevs)) >> optional(fwd("parse_many_call_suffix"))

def correct_right_recursion(types, cls, lhs, rhs):
    """Corrects the output of parse_expr; It's a right-recursive grammar, which incorrectly groups expressions
    like 1 + 2 - 3 into 1 + (2 - 3)"""
    if type(rhs) in types:
        return type(rhs)(correct_right_recursion(types, cls, lhs, rhs.lhs), rhs.rhs)
    return cls(lhs, rhs)

parse_func_param_list = Pipeline("func-param-list") >> consume({"LPAREN"}) >> ambig_lookahead(
    parse_comma_list(Pipeline("func-param-list[0]") >> ambig_lookahead(
        next_token({"NAME"}),
        Pipeline("try-vararg") >> consume({"ELLIPSIS"}) >> set_result(0, lambda prevs: None)
    ) >> set_result(1, lambda prevs: prevs[0][1] if prevs[0] else None)),
    Pipeline("func-param-list[1]") >> set_result(0, lambda _: ())  # TODO: Do something with vararg
) >> consume({"RPAREN"}) >> set_result(1, lambda prevs: tuple(i for i in prevs[0] if i is not None))

parse_atom_suffix = Pipeline("atom-suffix") >> ambig_lookahead(
    Pipeline("atom-suffix[0]") >> parse_call_suffix >> set_result(2, lambda prevs: Call(*prevs)) >> optional(fwd("parse_atom_suffix")),
    Pipeline("atom-suffix[1]") >> consume({"LBRACKET"}) >> fwd("parse_expr") >> consume({"RBRACKET"}) >> set_result(2, lambda prevs: Index(*prevs)) >> optional(fwd("parse_atom_suffix")),
    Pipeline("atom-suffix[2]") >> consume({"DOT"}) >> next_token({"NAME"}) >> set_result(2, lambda prevs: Index(prevs[0], String(prevs[1][1]))) >> optional(fwd("parse_atom_suffix")),
)
parse_atom = ambig_lookahead(
    parse_number,
    parse_string,
    Pipeline("neg") >> consume({"DASH"}) >> fwd("parse_atom") >> set_result(1, lambda prevs: Negative(prevs[0])),
    Pipeline("func") >> consume({"FUNCTION"}) >> parse_func_param_list >> fwd("parse_function_block") >> set_result(2, lambda prevs: InlineFunction(prevs[0], prevs[1])),
    Pipeline("getn") >> consume({"HASH"}) >> fwd("parse_atom") >> set_result(1, lambda prevs: Length(prevs[0])),  # TODO: This should be wayy more restrictive
    Pipeline("nil") >> consume({"NIL"}) >> set_result(0, lambda _: SpecialNil()),
    Pipeline("true") >> consume({"TRUE"}) >> set_result(0, lambda _: SpecialTrue()),
    Pipeline("vararg") >> consume({"ELLIPSIS"}) >> set_result(0, lambda _: Vararg()),
    Pipeline("false") >> consume({"FALSE"}) >> set_result(0, lambda _: SpecialFalse()),
    Pipeline("paren-atom") >> consume({"LPAREN"}) >> fwd("parse_expr") >> consume({"RPAREN"}) >> optional_lookahead(parse_atom_suffix) >> set_result(1, lambda prevs: Parenthesized(prevs[0])),
    Pipeline("name-atom") >> parse_name >> optional_lookahead(parse_atom_suffix),
    Pipeline("table") >> consume({"LCURLY"}) >> ambig_lookahead(
        Pipeline("table-content") >> parse_comma_list(
            ambig(
                ambig_lookahead(
                    Pipeline("table-named-elem") >> next_token({"NAME"}) >> consume({"EQUALS"}) >> set_result(1, lambda prevs: String(prevs[0][1])),
                    Pipeline("table-exprd-elem") >> next_token({"LBRACKET"}) >> fwd("parse_expr") >> consume({"RBRACKET"}),
                ) >> fwd("parse_expr") >> set_result(2, lambda prevs: prevs),
                fwd("parse_expr")
            )
        ),
        Pipeline("table-empty") >> set_result(0, lambda _: ())
    ) >> consume({"RCURLY"}) >> set_result(1, lambda prevs: Table(prevs[0]))
)

# Operator precedence logic
parse_muldiv = Pipeline("muldiv") >> parse_atom >> optional_lookahead(ambig_lookahead(
    Pipeline("mul") >> consume({"STAR"})  >> fwd("parse_muldiv") >> set_result(2, lambda sides: correct_right_recursion((Multiply, Divide), Multiply, sides[0], sides[1])),
    Pipeline("div") >> consume({"SLASH"}) >> fwd("parse_muldiv") >> set_result(2, lambda sides: correct_right_recursion((Multiply, Divide), Divide,   sides[0], sides[1])),
))
parse_addsub = Pipeline("addsub") >> parse_muldiv >> optional_lookahead(ambig_lookahead(
    Pipeline("add") >> consume({"PLUS"}) >> fwd("parse_addsub") >> set_result(2, lambda sides: correct_right_recursion((Add, Subtract), Add,      sides[0], sides[1])),
    Pipeline("sub") >> consume({"DASH"}) >> fwd("parse_addsub") >> set_result(2, lambda sides: correct_right_recursion((Add, Subtract), Subtract, sides[0], sides[1]))
))
parse_concat = Pipeline("concat") >> parse_addsub >> optional_lookahead(
    Pipeline("concat-sub") >> consume({"CONCAT"}) >> fwd("parse_concat") >> set_result(2, lambda sides: correct_right_recursion((Concat,), Concat, sides[0], sides[1]))
)
parse_comparison = Pipeline("comparison") >> parse_concat >> optional_lookahead(ambig_lookahead(
    Pipeline("lt") >> consume({"LT"})  >> fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), Less,         sides[0], sides[1])),
    Pipeline("gt") >> consume({"GT"})  >> fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), Greater,      sides[0], sides[1])),
    Pipeline("le") >> consume({"LEQ"}) >> fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), LessEqual,    sides[0], sides[1])),
    Pipeline("ge") >> consume({"GEQ"}) >> fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), GreaterEqual, sides[0], sides[1])),
    Pipeline("eq") >> consume({"DEQ"}) >> fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), Equal,        sides[0], sides[1])),
    Pipeline("ne") >> consume({"TEQ"}) >> fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), NotEqual,     sides[0], sides[1])),
))
parse_expr = Pipeline("expr") >> parse_comparison  # The expression parser

## Top level construct parsers

parse_variable_suffix_or_call_suffix = Pipeline("varsuffix") >> ambig_lookahead(
    Pipeline("varsuffix-attr") >> ambig_lookahead(
        Pipeline("varsuffix-attr-name") >> consume({"DOT"}) >> next_token({"NAME"}) >> set_result(1, lambda prevs: String(prevs[0][1])),
        Pipeline("varsuffix-attr-expr") >> consume({"LBRACKET"}) >> parse_expr >> consume({"RBRACKET"})
    ) >> set_result(2, lambda prevs: Index(*prevs)) >> optional(fwd("parse_variable_suffix_or_call_suffix")),
    Pipeline("varsuffix-call") >> parse_call_suffix >> set_result(2, lambda prevs: Call(*prevs)) >> optional(fwd("parse_variable_suffix_or_call_suffix"))
)
# TODO: The grammar is too broad and accepts expressions like f() = ...
parse_varlist_or_call = Pipeline("varlist") >> ambig_lookahead(
    Pipeline("varlist-parenexpr") >> parse_surrounded(fwd("parse_expr"), "LPAREN", "RPAREN") >> parse_variable_suffix_or_call_suffix,
    Pipeline("varlist-name") >> parse_name >> optional_lookahead(parse_variable_suffix_or_call_suffix),
) >> ambig(
    Pipeline("varlist-rest") >> consume({"COMMA"}) >> (lambda toks, prevs: ErrorParserState(toks, toks.toks[toks.idx], frozenset()) if isinstance(prevs[-1], Call) else OKParserState(toks, prevs)) >> fwd("parse_varlist_or_call"),
    set_result(0, lambda _: ())
) >> set_result(2, lambda prevs: prevs[0] if isinstance(prevs[0], Call) else (prevs[0], *prevs[1]))

parse_function_block = Pipeline("func-block") >> fwd("parse_end_block")

inspect = Pipeline("inspect") >> (lambda toks, res: (print(res), OKParserState(toks, res))[1])

parse_statement = Pipeline("stmt") >> ambig_lookahead(
    Pipeline("local") >> consume({"LOCAL"}) >> ambig_lookahead(
        Pipeline("local-func") >> consume({"FUNCTION"}) >> next_token({"NAME"}) >> set_result(1, lambda prevs: prevs[0][1]) >> parse_func_param_list >> parse_function_block >> set_result(3, lambda prevs: LocalFunctionDef(*prevs)),
        Pipeline("local-rest") >> parse_comma_list(Pipeline("local-namelist") >> next_token({"NAME"}) >> set_result(1, lambda prevs: prevs[0][1])) >> ambig_lookahead(
            Pipeline("local-asgn") >> consume({"EQUALS"}) >> parse_expr >> set_result(2, lambda prevs: LocalAssign(prevs[0], prevs[1])),
            Pipeline("local-decl") >> set_result(1, lambda prevs: Locals(prevs[0]))
        )
    ),

    Pipeline("func") >> consume({"FUNCTION"}) >> parse_name >> optional_lookahead(
        Pipeline("func-namelist") >> consume({"DOT"}) >> parse_sep_list(Pipeline >> next_token({"NAME"}) >> set_result(2, lambda prevs: Index(prevs[0], String(prevs[1][1]))), "DOT")
    ) >> ambig_lookahead(
        Pipeline("func-selfarg") >> next_token({"COLON"}) >> next_token({"NAME"}) >> set_result(2, lambda prevs: Index(prevs[0], String(prevs[1][1]))) >> set_result(0, lambda _: SelfFunctionDef),
        Pipeline("func-no-selfarg") >> set_result(0, lambda _: FunctionDef)
    ) >> parse_func_param_list >> parse_function_block >> set_result(4, lambda prevs: prevs[1](prevs[0], prevs[2], prevs[3])),
    Pipeline("if") >> consume({"IF"}) >> parse_expr >> consume({"THEN"}) >> ambig(
        Pipeline("if-end") >> fwd("parse_end_block") >> set_result(2, lambda prevs: If(prevs[0], prevs[1])),
        Pipeline("if-else-end") >> fwd("parse_else_block") >> fwd("parse_end_block") >> set_result(3, lambda prevs: IfElse(prevs[0], prevs[1], prevs[2])),
    ),

    Pipeline("while") >> consume({"WHILE"}) >> parse_expr >> consume({"DO"}) >> fwd("parse_end_block") >> set_result(2, lambda prevs: While(prevs[0], prevs[1])),
    Pipeline("for") >> consume({"FOR"}) >> next_token({"NAME"}) >> consume({"EQUALS"}) >> parse_expr >> consume({"COMMA"}) >> parse_expr >> ambig_lookahead(
        consume({"COMMA"}) >> parse_expr, set_result(0, lambda _: None)
    ) >> consume({"DO"}) >> fwd("parse_end_block") >> set_result(5, lambda prevs: For(prevs[0][1], *prevs[1:])),

    Pipeline("return") >> consume({"RETURN"}) >> ambig_lookahead(parse_comma_list(parse_expr), set_result(0, lambda _: ())) >> set_result(1, lambda prevs: Return(prevs[0])),
    Pipeline("break") >> consume({"BREAK"}) >> set_result(0, lambda _: Break()),

    Pipeline("asgn-or-call") >> parse_varlist_or_call >> optional_lookahead(
        consume({"EQUALS"}) >> (lambda toks, prevs: ErrorParserState(toks, toks.toks[toks.idx], frozenset()) if isinstance(prevs[-1], Call) else OKParserState(toks, prevs)) >> parse_expr >> set_result(2, lambda prevs: Assign(prevs[0], prevs[1]))
    )
)

parse_block_inner = Pipeline("block-inner") >> set_result(0, lambda _: ()) >> parse_list(parse_statement >> set_result(2, lambda prevs: (*prevs[0], prevs[1])))
parse_block = lambda end_toks: Pipeline("block") >> ambig(parse_block_inner, set_result(0, lambda _: ())) >> consume(end_toks) >> set_result(1, lambda prevs: Block(prevs[0]))
parse_end_block = parse_block({"END"})
parse_else_block = parse_block({"ELSE"})
parse_chunk = Pipeline("chunk") >> parse_block({"EOF"}) >> set_result(1, lambda prevs: Chunk(prevs[0]))  # Parsing of a program starts here

## End of parser code
_Pipeline.resolve_fwd()

## This is the only function that should be used from this library
def parse(tokens):
    match parse_chunk(tokens, ()):
        case OKParserState(_, (tree,)):
            return opcodes.encode(tree.code)
        case ErrorParserState(_, pos, expected, _):
            raise ParseError(pos, f"expected one of {', '.join(expected)}")
    raise RuntimeError("Parser somehow returned multiple results")

def main():
    import time

    builtin = ["basic.lua", "table.lua", "test.lua"]
    toklists = []
    start = time.perf_counter()
    for filename in builtin:
        with open(filename, encoding="ascii") as f:
            toklists.append(Tokens.from_string(f.read(), filename))
            print("done lex", filename)
    print(f"{1000 * (time.perf_counter() - start):.2f} ms")
    print("total tokens", sum(len(t.toks) for t in toklists))

    start = time.perf_counter()
    for tokens in toklists:
        parse(tokens)
        print("done parse", filename)
    print(f"{1000 * (time.perf_counter() - start):.2f} ms")

if __name__ == "__main__":
    main()
