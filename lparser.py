# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

import inspect  # TODO: Integarte the changes from the transpiling branch here (this code is still very ugly)
from typing import Any
from dataclasses import dataclass

import opcodes
from llexer import ParseError, Tokens

DEBUG = False

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
    """Helper class for constructing parser pipelines"""

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
            if isinstance(step, Fwd):
                child = step.globals[step.name]
                if child not in stack:
                    child._resolve_fwd(stack)  # pylint: disable=protected-access
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
            pipeline._resolve_fwd([])  # pylint: disable=protected-access

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

class Fwd:

    @property
    def expect(self):
        return None

    def __call__(self, *args):
        # Not all Fwd objects will be resolved (some are recursive)
        return self.globals[self.name](*args)

    def __init__(self, name):
        self.name = name
        self.globals = inspect.currentframe().f_back.f_globals   # TODO: Ughhh horrifying really you need to get those changes over

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
def set_result(n, nodefunc):
    return _Pipeline([lambda toks, prevs: OKParserState(toks, (*(prevs[:-n] if n > 0 else prevs), nodefunc(prevs[-n:])))], "set_result", None)

# Constructs a parser that consumes a token, but doesn't do anything with it and doesn't modify the results list
def consume(expect=None):
    def wrap(toks, prevs):
        (pos, tok), toks = toks.advance()
        if expect is None or tok[0] in expect:
            return OKParserState(toks, prevs)
        return ErrorParserState(toks, pos, expect)
    return _Pipeline([wrap], "consume", frozenset(expect))

# Constructs a parser that consumes a token, invokes the `func` parser, and consumes another token
def parse_surrounded(func, left, right):
    return Pipeline(f"surrounded({func!r},{left!r},{right!r})") >> consume({left}) >> func >> consume({right})

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

def parse(grammar, tokens):
    match grammar(tokens, ()):
        case OKParserState(_, (tree,)):
            return opcodes.encode(tree.code)
        case ErrorParserState(_, pos, expected, _):
            raise ParseError(pos, f"expected one of {', '.join(expected)}")
    raise RuntimeError("Parser somehow returned multiple results")
