# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

import traceback
import collections
from contextlib import contextmanager
from dataclasses import dataclass, field

import llexer
import opcodes
from llexer import ParseError

ENABLE_PRINTNO = False
def printno(arg=None):
    if ENABLE_PRINTNO:
        where = f"> {arg}" if arg else traceback.format_stack()[-2].strip().split("\n")[0].split("in ")[-1]
        print(" " * len(traceback.format_stack()), where)

class Label:
    pass

class StringRef(collections.UserString):
    __hash__ = None

    def set(self, data):
        self.data = data

@dataclass
class ClosureInfo:
    localvars: set = field(default_factory=set)

@dataclass
class FunctionClosureInfo(ClosureInfo):
    parent_closures: list = field(default_factory=list)

    def collect_upvalues(self):
        uvs = {j for info in self.parent_closures for j in info.localvars} | {j for i in self.parent_closures[-1:] for j in i.collect_upvalues()}
        return uvs | {"_ENV"}

class InstructionStream:

    def __init__(self):
        self.labels = {}
        self.instructions = []
        self.closure_infos = []

    @contextmanager
    def push_block(self, break_label: str | Label | None = "implicit", is_function: bool = False):
        if is_function:
            if break_label not in {None, "implicit"}:
                raise ValueError("functions can't have a break target")
            self.closure_infos.append(FunctionClosureInfo())
            for info in self.closure_infos[-2::-1]:
                self.closure_infos[-1].parent_closures.append(info)
                if isinstance(info, FunctionClosureInfo):
                    break
        else:
            self.push(opcodes.PushBlock(label := Label()) if break_label == "implicit" else opcodes.PushBlock(break_label or 0))
            self.closure_infos.append(ClosureInfo())
        if is_function:
            self.push(opcodes.UpvalueInfo(self.closure_infos[-1]))
        self.push(opcodes.BlockInfo(self.closure_infos[-1]))
        try:
            yield
        finally:
            self.closure_infos.pop()
            if not is_function:
                if break_label == "implicit":
                    self.push(opcodes.PopBlock())
                    self.push(label, opcodes.Nop())  # TODO: Get rid of this NOP somehow, it's bugging me
                else:
                    self.push(opcodes.PopBlock())

    @contextmanager
    def push_inline_block(self, push_inline_labels: list[Label] | None = None):
        opcode_offs = self.push(*(push_inline_labels or []), (opcode := opcodes.PushInline(0)))
        try:
            yield
        finally:
            opcode.arg = len(self.instructions) - opcode_offs - 1

    @contextmanager
    def push_stack_guard(self, top: int | None = None, push_labels: list[Label] | None = None, pop_labels: list[Label] | None = None):
        self.push(*(push_labels or []), opcodes.PushBound())
        try:
            yield
        finally:
            if top == 0:
                self.push(*(pop_labels or []), opcodes.PopBound())
            elif top is None:
                self.push(*(pop_labels or []), opcodes.DiscardBound())
            else:
                self.push(opcodes.SetTop(top))
                self.push(*(pop_labels or []), opcodes.DiscardBound())

    def set_labels(self, labels):
        for label in labels:
            if label in self.labels:
                raise ValueError(f"label {label} already has target")
            self.labels[label] = len(self.instructions)

    def push(self, *args):
        *labels, op = args
        self.set_labels(labels)
        self.instructions.append(op)
        return len(self.instructions) - 1

    def edit(self, at, *args):
        *labels, op = args
        self.set_labels(labels)
        self.instructions[at] = op

    def serialize(self):
        strings = []
        def add_string(s):
            try:
                return strings.index(s)
            except ValueError:
                if len(strings) == 9999:
                    raise ValueError("chunk has over 9999 strings") from None
                strings.append(s)
                return len(strings) - 1

        result = ""
        for op in self.instructions:
            arg = op.arg
            if isinstance(arg, str):
                arg = add_string(arg)

            elif isinstance(arg, ClosureInfo):
                if isinstance(op, opcodes.UpvalueInfo):  # a bit dirty but whatever
                    assert isinstance(arg, FunctionClosureInfo)
                    info = "".join(str(add_string(i)).zfill(4) for i in arg.collect_upvalues())
                else:
                    info = "".join(str(add_string(i)).zfill(4) for i in arg.localvars)
                arg = add_string(info)

            elif isinstance(arg, Label):
                arg = self.labels[arg]

            if not 0 <= arg < 9999:
                raise ValueError(f"{op} arg(={arg}) out of range")
            result += op.op + str(arg).zfill(4)
        return "$" + "".join(f"{len(string):04d}{string}" for string in strings) + ";" + result

    def add_local(self, instream: llexer.TokenStream, name: str, allow_redefine: bool = False):
        if name in self.closure_infos[-1].localvars:
            if allow_redefine:
                return
            raise ParseError(instream.pos, "variable declared as local twice")
        self.closure_infos[-1].localvars.add(name)

def parse_statement(instream: llexer.TokenStream, outstream: InstructionStream):
    printno()
    tok, data = instream.next()
    if tok in {"NAME", "LPAREN"}:
        instream.rewind()
        outstream.push(opcodes.PushBound())
        typ = parse_var(instream, outstream, use_ref=True)
        tok, data = instream.next()
        if tok in {"EQUALS", "COMMA"}:
            outstream.push(opcodes.DiscardBound())
            nvars = 1
            while True:
                if tok == "EQUALS":
                    break
                parse_var(instream, outstream, use_ref=True)
                nvars += 1
                tok, _ = instream.next(expect={"COMMA", "EQUALS"})

            with outstream.push_stack_guard(top=nvars):
                parse_expr(instream, outstream)
            outstream.push(opcodes.SetDual(nvars))
            return
        if typ != "call":
            raise ParseError(instream.pos, "function call or assignment expected")

        instream.rewind()
        outstream.push(opcodes.PopBound())
        return

    if tok == "SEMICOLON":
        return

    if tok == "BREAK":
        outstream.push(opcodes.Break())
        return

    if tok == "DO":
        parse_block(instream, outstream, {"END"})
        return

    if tok == "FOR":
        # TODO: Iterator for
        _, name = instream.next(expect={"NAME"})
        with outstream.push_block(block_return_label := Label()):
            outstream.push(opcodes.PushBound())
            outstream.push(opcodes.RefName(name))
            instream.next(expect={"EQUALS"})
            outstream.add_local(instream, name, allow_redefine=True)
            with outstream.push_stack_guard(top=1):
                parse_expr(instream, outstream)
            instream.next(expect={"COMMA"})
            outstream.push(opcodes.SetDual(1))

            with outstream.push_stack_guard(top=1):
                parse_expr(instream, outstream)

            tok, _ = instream.next(expect={"DO", "COMMA"})
            if tok == "COMMA":
                with outstream.push_stack_guard(top=1):
                    parse_expr(instream, outstream)
                instream.next(expect={"DO"})
            else:
                outstream.push(opcodes.PushNumber("1"))

            with outstream.push_stack_guard(top=1, push_labels=[jump_target := Label()]):
                outstream.push(opcodes.Dup(1))
                outstream.push(opcodes.Sign())
                outstream.push(opcodes.Dup(0))
                outstream.push(opcodes.Dup(4))
                outstream.push(opcodes.Multiply())
                outstream.push(opcodes.GetName(name))
                outstream.push(opcodes.Dup(2))
                outstream.push(opcodes.Multiply())
                outstream.push(opcodes.GreaterEqual())
                outstream.push(opcodes.Swap())

            with outstream.push_inline_block():
                parse_block(instream, outstream, {"END"})

                outstream.push(opcodes.RefName(name))
                outstream.push(opcodes.GetName(name))
                outstream.push(opcodes.Dup(2))
                outstream.push(opcodes.Add())
                outstream.push(opcodes.SetDual(1))
                outstream.push(opcodes.Jump(jump_target))

            outstream.push(opcodes.If())

        outstream.push(block_return_label, opcodes.PopBound())
        return

    if tok == "WHILE":
        with outstream.push_block():
            with outstream.push_stack_guard(top=1, push_labels=[target := Label()]):
                parse_expr(instream, outstream)

            instream.next(expect={"DO"})

            with outstream.push_inline_block():
                parse_block(instream, outstream, {"END"})
                outstream.push(opcodes.Jump(target))

            outstream.push(opcodes.If())
        return

    if tok == "IF":
        # TODO: Separate block for each clause
        with outstream.push_block(break_label=None):
            parse_if(instream, outstream)
        return

    if tok == "FUNCTION":
        parse_tl_function(instream, outstream, local=False)
        return

    if tok == "LOCAL":
        tok, data = instream.next()
        if tok == "FUNCTION":
            parse_tl_function(instream, outstream, local=True)
            return

        if tok == "NAME":
            name = data
            outstream.add_local(instream, name)
            tok, data = instream.next()
            # TODO: Multiple local assignment
            if tok != "EQUALS":
                # end of statement
                instream.rewind()
                return
            outstream.push(opcodes.RefName(name))
            with outstream.push_stack_guard(top=1):
                parse_expr(instream, outstream)
            outstream.push(opcodes.SetDual(1))
            return

        raise ParseError(instream.pos, f"expected FUNCTION or NAME, got {tok}")

    if tok == "RETURN":
        outstream.push(opcodes.PushBound())
        outstream.push(opcodes.PushBound())
        if parse_expr(instream, outstream, optional=True):
            while True:
                tok, data = instream.next()
                if tok != "COMMA":
                    instream.rewind()
                    outstream.push(opcodes.DiscardBound())
                    break
                outstream.push(opcodes.SetTop(1))
                outstream.push(opcodes.DiscardBound())
                outstream.push(opcodes.PushBound())
                parse_expr(instream, outstream)
            outstream.push(opcodes.Return(0))
        return

    raise ParseError(instream.pos, f"unexpected token {tok}")

def parse_if(instream: llexer.TokenStream, outstream: InstructionStream):
    printno()
    with outstream.push_stack_guard(top=1):
        parse_expr(instream, outstream)

    instream.next(expect={"THEN"})
    with outstream.push_inline_block():
        parse_block(instream, outstream, {"END", "ELSE", "ELSEIF"})
        outstream.push(opcodes.Jump(jump1 := Label()))

    instream.rewind()
    tok, _ = instream.next()
    if tok in {"ELSE", "ELSEIF"}:
        with outstream.push_inline_block():
            if tok == "ELSEIF":
                parse_if(instream, outstream)
            else:
                parse_block(instream, outstream, {"END", "ELSE"})
            outstream.push(opcodes.Jump(jump2 := Label()))
        outstream.push(opcodes.IfElse())
        outstream.push(jump1, jump2, opcodes.Nop())
    else:
        outstream.push(opcodes.If())
        outstream.push(jump1, opcodes.Nop())

def parse_anon_function(instream: llexer.TokenStream, outstream: InstructionStream):
    printno()
    instream.next(expect={"LPAREN"})
    params = []
    while True:
        tok, data = instream.next(expect={"NAME", "RPAREN"})
        if tok == "RPAREN":
            break

        params.append(data)
        tok, data = instream.next(expect={"COMMA", "RPAREN"})
        if tok == "RPAREN":
            break

    with outstream.push_inline_block():
        parse_block(instream, outstream, is_function=True, locals_=set(params))
        outstream.push(opcodes.PushBound())
        outstream.push(opcodes.Return(0))

    for i in params:
        outstream.push(opcodes.PushString(i))
    outstream.push(opcodes.PushFunction(len(params)))

def parse_tl_function(instream: llexer.TokenStream, outstream: InstructionStream, /, local):
    printno()
    tok, data = instream.next(expect={"NAME"})
    path = []
    selfarg = False
    while True:
        path.append(data)
        if len(path) > 1 and local:
            raise ParseError(instream.pos, "table function can't be declared local")
        tok, data = instream.next(expect={"LPAREN", "COLON", "DOT"})
        if tok == "LPAREN":
            break
        if tok == "COLON":
            tok, data = instream.next(expect={"NAME"})
            selfarg = True
            path.append(data)
            break
        tok, data = instream.next(expect={"NAME"})

    params = []
    if selfarg:
        params.append("self")
    while True:
        tok, data = instream.next(expect={"RPAREN", "NAME", "ELLIPSIS"})
        if tok == "RPAREN":
            break
        if tok == "NAME":
            params.append(data)
        tok, data = instream.next(expect={"RPAREN", "COMMA"})
        if tok == "RPAREN":
            break
    name = path.pop()

    if path:
        outstream.push(opcodes.GetName(path[0]))
        for seg in path[1:]:
            outstream.push(opcodes.PushString(seg))
            outstream.push(opcodes.GetItem())
        outstream.push(opcodes.PushString(name))
        outstream.push(opcodes.RefItem())
    else:
        if local:
            outstream.add_local(instream, name)
        outstream.push(opcodes.RefName(name))

    with outstream.push_inline_block():
        parse_block(instream, outstream, is_function=True, locals_=set(params))
        outstream.push(opcodes.PushBound())
        outstream.push(opcodes.Return(0))
    for i in params:
        outstream.push(opcodes.PushString(i))
    outstream.push(opcodes.PushFunction(len(params)))

    outstream.push(opcodes.SetDual(1))

def parse_block(instream: llexer.TokenStream, outstream: InstructionStream, end_tokens=("END",), is_function=False, locals_: set | None = None):
    printno()
    with outstream.push_block(break_label=None, is_function=is_function):
        for local in locals_ or {}:
            outstream.add_local(instream, local)
        while True:
            tok, _ = instream.next()
            if tok in end_tokens:
                return
            instream.rewind()
            parse_statement(instream, outstream)

def parse_var(instream: llexer.TokenStream, outstream: InstructionStream, use_ref: bool = False):
    printno()
    tok, data = instream.next(expect={"NAME", "LPAREN"})
    if tok == "NAME":
        outstream.push(opcodes.GetName(data))
        tok, data = instream.next()
    elif tok == "LPAREN":
        parse_expr(instream, outstream)
        tok, data = instream.next()
        if tok != "RPAREN":
            raise ParseError(instream.pos, f"expected RPAREN, got {tok}")
        tok, data = instream.next()

    typ = "var"
    while True:
        if tok == "LBRACKET":
            typ = "var"
            parse_expr(instream, outstream)
            instream.next(expect={"RBRACKET"})
            outstream.push(opcodes.GetItem())

        elif tok == "DOT":
            typ = "var"
            tok, data = instream.next(expect={"NAME"})
            outstream.push(opcodes.PushString(data))
            outstream.push(opcodes.GetItem())

        elif tok == "QSTR":
            typ = "call"
            outstream.push(opcodes.PushBound())
            outstream.push(opcodes.PushString(data))
            outstream.push(opcodes.Call(0))

        elif tok == "LCURLY":
            typ = "call"
            outstream.push(opcodes.PushBound())
            parse_table(instream, outstream)
            outstream.push(opcodes.Call(0))

        elif tok == "LPAREN":
            typ = "call"
            parse_call(instream, outstream)

        else:
            instream.rewind()
            if isinstance(outstream.instructions[-1], opcodes.GetName) and use_ref:
                outstream.edit(-1, opcodes.RefName(outstream.instructions[-1].arg))
            elif isinstance(outstream.instructions[-1], opcodes.GetItem) and use_ref:
                outstream.edit(-1, opcodes.RefItem())
            return typ

        tok, data = instream.next()

    if isinstance(outstream.instructions[-1], opcodes.GetName) and use_ref:
        outstream.edit(-1, opcodes.RefName(outstream.instructions[-1].arg))
    return "var"

def parse_call(instream: llexer.TokenStream, outstream: InstructionStream, limit: int | None = -1):
    printno()
    outstream.push(opcodes.PushBound())
    tok, _ = instream.next()
    if tok == "RPAREN":
        outstream.push(opcodes.Call(0))
        return
    instream.rewind()
    while True:
        outstream.push(opcodes.PushBound())
        parse_expr(instream, outstream)
        tok, _ = instream.next(expect={"RPAREN", "COMMA"})
        if tok == "RPAREN":
            outstream.push(opcodes.DiscardBound())
            outstream.push(opcodes.Call(limit + 1))
            return
        outstream.push(opcodes.SetTop(1))
        outstream.push(opcodes.DiscardBound())

def parse_table(instream: llexer.TokenStream, outstream: InstructionStream):
    printno()
    tok, data = instream.next()
    if tok == "RCURLY":
        outstream.push(opcodes.PushBound())
        outstream.push(opcodes.PushTable(0))
        return
    instream.rewind()

    n_pairs = 0
    while True:
        tok, data = instream.next()
        if tok == "NAME":
            name = data
            tok, data = instream.next()
            if tok != "COMMA":
                if tok != "EQUALS":
                    instream.rewind()
                else:
                    with outstream.push_stack_guard(top=2):
                        outstream.push(opcodes.PushString(name))
                        parse_expr(instream, outstream)
                    n_pairs += 1
                    tok, data = instream.next(expect={"RCURLY", "COMMA"})
                    if tok == "RCURLY":
                        break
                    continue

        elif tok == "LBRACKET":
            with outstream.push_stack_guard(top=1):
                parse_expr(instream, outstream)
            instream.next(expect={"RBRACKET"})
            instream.next(expect={"EQUALS"})
            with outstream.push_stack_guard(top=1):
                parse_expr(instream, outstream)
            n_pairs += 1
            tok, data = instream.next(expect={"RCURLY", "COMMA"})
            if tok == "RCURLY":
                break
            continue

        instream.rewind()
        outstream.push(opcodes.PushBound())
        parse_expr(instream, outstream)

        tok, data = instream.next(expect={"RCURLY", "COMMA"})
        if tok == "RCURLY":
            outstream.push(opcodes.PushTable(n_pairs))
            return

        outstream.push(opcodes.SetTop(1))
        outstream.push(opcodes.DiscardBound())
        outstream.push(opcodes.PushNil())
        outstream.push(opcodes.Swap())
        n_pairs += 1

    outstream.push(opcodes.PushBound())
    outstream.push(opcodes.PushTable(n_pairs))

PRECEDENCE = {"OR": -2, "AND": -1, "DEQ": 0, "LT": 0, "GT": 0, "LEQ": 0, "GEQ": 0, "TEQ": 0, "CONCAT": 1, "PLUS": 2, "DASH": 2, "STAR": 3, "SLASH": 3}
OPMAP = {"PLUS": opcodes.Add(), "DASH": opcodes.Subtract(), "STAR": opcodes.Multiply(), "SLASH": opcodes.Divide(),
         "UNARYDASH": opcodes.Negative(), "UNARYHASH": opcodes.Len(), "DEQ": opcodes.Equal(), "LT": opcodes.Less(),
         "GT": opcodes.Greater(), "LEQ": opcodes.LessEqual(), "GEQ": opcodes.GreaterEqual(), "TEQ": opcodes.Unequal(), "OR": opcodes.Or(),
         "AND": opcodes.And(), "UNARYNOT": opcodes.Not(), "CONCAT": opcodes.Concat()}
UNARY_OP = {"DASH": "UNARYDASH", "HASH": "UNARYHASH", "NOT": "UNARYNOT"}
def parse_expr(instream: llexer.TokenStream, outstream: InstructionStream, optional: bool = False):
    printno()
    def pop_op_stack():
        tok = op_stack.pop()
        outstream.push(OPMAP[tok])

    def pop_unary_ops():
        while op_stack and op_stack[-1] in UNARY_OP.values():
            tok = op_stack.pop()
            outstream.push(OPMAP[tok])

    op_stack = []
    was_prev_value = False
    parsed_anything = False
    multi_repl = None
    this_multi = False
    while True:
        this_multi = False
        tok, data = instream.next()
        if tok == "QSTR":
            if was_prev_value:
                if multi_repl is not None and not this_multi:
                    outstream.edit(multi_repl, opcodes.SetTop(1))
                    multi_repl = None
                outstream.push(opcodes.PushBound())
                outstream.push(opcodes.PushString(data))
                outstream.push(opcodes.PushBound())
                outstream.push(opcodes.Call(data))
                this_multi = True
                multi_repl = outstream.push(opcodes.Nop())
                outstream.push(opcodes.DiscardBound())
                pop_unary_ops()
            else:
                outstream.push(opcodes.PushString(data))
                pop_unary_ops()
                was_prev_value = True
        elif tok == "ELLIPSIS":
            if was_prev_value:
                instream.rewind()
                break
            outstream.push(opcodes.PushBound())
            outstream.push(opcodes.PushVarargs())
            this_multi = True
            multi_repl = outstream.push(opcodes.Nop())
            outstream.push(opcodes.DiscardBound())
            pop_unary_ops()
            was_prev_value = True
        elif tok == "NUMBER":
            if was_prev_value:
                instream.rewind()
                break
            outstream.push(opcodes.PushNumber(data))
            pop_unary_ops()
            was_prev_value = True
        elif tok == "NIL":
            if was_prev_value:
                instream.rewind()
                break
            outstream.push(opcodes.PushNil())
            pop_unary_ops()
            was_prev_value = True
        elif tok == "TRUE":
            if was_prev_value:
                instream.rewind()
                break
            outstream.push(opcodes.PushTrue())
            pop_unary_ops()
            was_prev_value = True
        elif tok == "FALSE":
            if was_prev_value:
                instream.rewind()
                break
            outstream.push(opcodes.PushFalse())
            pop_unary_ops()
            was_prev_value = True
        elif tok == "LPAREN":
            if was_prev_value:
                parse_call(instream, outstream, limit=1)
            else:
                parse_expr(instream, outstream)
                pop_unary_ops()
                instream.next(expect={"RPAREN"})
                was_prev_value = True
        elif tok == "NAME":
            if was_prev_value:
                instream.rewind()
                break
            instream.rewind()
            outstream.push(opcodes.PushBound())
            parse_var(instream, outstream)
            this_multi = True
            multi_repl = outstream.push(opcodes.Nop())
            outstream.push(opcodes.DiscardBound())
            pop_unary_ops()
            was_prev_value = True
        elif tok == "FUNCTION":
            if was_prev_value:
                instream.rewind()
                break
            parse_anon_function(instream, outstream)
            pop_unary_ops()
            was_prev_value = True
        elif tok == "LCURLY":
            if was_prev_value:
                if multi_repl is not None and not this_multi:
                    outstream.edit(multi_repl, opcodes.SetTop(1))
                    multi_repl = None
                outstream.push(opcodes.PushBound(), 0)
                outstream.push(opcodes.PushBound(), 0)
                parse_table(instream, outstream)
                outstream.push(opcodes.Call(data))
                multi_repl = len(outstream.instructions)
                this_multi = True
                outstream.push(opcodes.Nop())
                outstream.push(opcodes.DiscardBound())
                pop_unary_ops()
            else:
                parse_table(instream, outstream)
                pop_unary_ops()
                was_prev_value = True
        elif tok in UNARY_OP and not was_prev_value:
            op_stack.append(UNARY_OP[tok])
        elif tok in PRECEDENCE:
            if not was_prev_value:
                # TODO: Is this correct?
                raise ParseError(instream.pos, "unexpected operator after operator")
            while op_stack:
                if op_stack[-1] in UNARY_OP.values():
                    raise ParseError(instream.pos, "unexpected operator after unary operator")
                if PRECEDENCE[op_stack[-1]] < PRECEDENCE[tok]:
                    break
                pop_op_stack()
            op_stack.append(tok)
            was_prev_value = False
        else:
            instream.rewind()
            break
        if multi_repl is not None and not this_multi:
            outstream.edit(multi_repl, opcodes.SetTop(1))
            multi_repl = None
        parsed_anything = True
    if not parsed_anything:
        if optional:
            return True
        raise ParseError(instream.pos, f"expected expression, got {tok}")
    while op_stack:
        pop_op_stack()
    return True

def parse_chunk(instream: llexer.TokenStream, outstream: InstructionStream):
    with outstream.push_inline_block():
        parse_block(instream, outstream, {"EOF"}, is_function=True)
        outstream.push(opcodes.PushBound())
        outstream.push(opcodes.Return(0))
    outstream.push(opcodes.PushFunction(0))
