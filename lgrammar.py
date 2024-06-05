# pylint: disable=line-too-long,unnecessary-lambda-assignment,wildcard-import,unused-wildcard-import

"""Contains the grammar for Lua"""

from last import *
from lparser import *
from lparser import _Pipeline

## Single token parsers

parse_name   = Pipeline("name")   >> next_token({"NAME"})   >> set_result(1, lambda prevs: NameRef(prevs[0][1]))
parse_string = Pipeline("string") >> next_token({"QSTR"})   >> set_result(1, lambda prevs: String(prevs[0][1]))
parse_number = Pipeline("number") >> next_token({"NUMBER"}) >> set_result(1, lambda prevs: Number(prevs[0][1]))

## Expression/subexpression parsers

parse_paren_expr_list = Pipeline("paren_expr_list") >> parse_surrounded(parse_comma_list(Fwd("parse_expr")), "LPAREN", "RPAREN")
parse_call_suffix = ambig_lookahead(
    Pipeline("call-suffix[0]") >> consume({"LPAREN"}) >> consume({"RPAREN"}) >> set_result(0, lambda _: ()),
    Pipeline("call-suffix[1]") >> parse_string >> set_result(1, lambda prevs: (prevs[0],)),
    parse_paren_expr_list
)
parse_many_call_suffix = Pipeline("call-suffix*") >> parse_call_suffix >> set_result(2, lambda prevs: Call(*prevs)) >> optional(Fwd("parse_many_call_suffix"))

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
    Pipeline("atom-suffix[0]") >> parse_call_suffix >> set_result(2, lambda prevs: Call(*prevs)) >> optional(Fwd("parse_atom_suffix")),
    Pipeline("atom-suffix[1]") >> consume({"LBRACKET"}) >> Fwd("parse_expr") >> consume({"RBRACKET"}) >> set_result(2, lambda prevs: Index(*prevs)) >> optional(Fwd("parse_atom_suffix")),
    Pipeline("atom-suffix[2]") >> consume({"DOT"}) >> next_token({"NAME"}) >> set_result(2, lambda prevs: Index(prevs[0], String(prevs[1][1]))) >> optional(Fwd("parse_atom_suffix")),
)
parse_atom = ambig_lookahead(
    parse_number,
    parse_string,
    Pipeline("neg") >> consume({"DASH"}) >> Fwd("parse_atom") >> set_result(1, lambda prevs: Negative(prevs[0])),
    Pipeline("func") >> consume({"FUNCTION"}) >> parse_func_param_list >> Fwd("parse_function_block") >> set_result(2, lambda prevs: InlineFunction(prevs[0], prevs[1])),
    Pipeline("getn") >> consume({"HASH"}) >> Fwd("parse_atom") >> set_result(1, lambda prevs: Length(prevs[0])),  # TODO: This should be wayy more restrictive
    Pipeline("nil") >> consume({"NIL"}) >> set_result(0, lambda _: SpecialNil()),
    Pipeline("true") >> consume({"TRUE"}) >> set_result(0, lambda _: SpecialTrue()),
    Pipeline("vararg") >> consume({"ELLIPSIS"}) >> set_result(0, lambda _: Vararg()),
    Pipeline("false") >> consume({"FALSE"}) >> set_result(0, lambda _: SpecialFalse()),
    Pipeline("paren-atom") >> consume({"LPAREN"}) >> Fwd("parse_expr") >> consume({"RPAREN"}) >> optional_lookahead(parse_atom_suffix) >> set_result(1, lambda prevs: Parenthesized(prevs[0])),
    Pipeline("name-atom") >> parse_name >> optional_lookahead(parse_atom_suffix),
    Pipeline("table") >> consume({"LCURLY"}) >> ambig_lookahead(
        Pipeline("table-content") >> parse_comma_list(
            ambig(
                ambig_lookahead(
                    Pipeline("table-named-elem") >> next_token({"NAME"}) >> consume({"EQUALS"}) >> set_result(1, lambda prevs: String(prevs[0][1])),
                    Pipeline("table-exprd-elem") >> next_token({"LBRACKET"}) >> Fwd("parse_expr") >> consume({"RBRACKET"}),
                ) >> Fwd("parse_expr") >> set_result(2, lambda prevs: prevs),
                Fwd("parse_expr")
            )
        ),
        Pipeline("table-empty") >> set_result(0, lambda _: ())
    ) >> consume({"RCURLY"}) >> set_result(1, lambda prevs: Table(prevs[0]))
)

# Operator precedence logic
parse_muldiv = Pipeline("muldiv") >> parse_atom >> optional_lookahead(ambig_lookahead(
    Pipeline("mul") >> consume({"STAR"})  >> Fwd("parse_muldiv") >> set_result(2, lambda sides: correct_right_recursion((Multiply, Divide), Multiply, sides[0], sides[1])),
    Pipeline("div") >> consume({"SLASH"}) >> Fwd("parse_muldiv") >> set_result(2, lambda sides: correct_right_recursion((Multiply, Divide), Divide,   sides[0], sides[1])),
))
parse_addsub = Pipeline("addsub") >> parse_muldiv >> optional_lookahead(ambig_lookahead(
    Pipeline("add") >> consume({"PLUS"}) >> Fwd("parse_addsub") >> set_result(2, lambda sides: correct_right_recursion((Add, Subtract), Add,      sides[0], sides[1])),
    Pipeline("sub") >> consume({"DASH"}) >> Fwd("parse_addsub") >> set_result(2, lambda sides: correct_right_recursion((Add, Subtract), Subtract, sides[0], sides[1]))
))
parse_concat = Pipeline("concat") >> parse_addsub >> optional_lookahead(
    Pipeline("concat-sub") >> consume({"CONCAT"}) >> Fwd("parse_concat") >> set_result(2, lambda sides: correct_right_recursion((Concat,), Concat, sides[0], sides[1]))
)
parse_comparison = Pipeline("comparison") >> parse_concat >> optional_lookahead(ambig_lookahead(
    Pipeline("lt") >> consume({"LT"})  >> Fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), Less,         sides[0], sides[1])),
    Pipeline("gt") >> consume({"GT"})  >> Fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), Greater,      sides[0], sides[1])),
    Pipeline("le") >> consume({"LEQ"}) >> Fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), LessEqual,    sides[0], sides[1])),
    Pipeline("ge") >> consume({"GEQ"}) >> Fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), GreaterEqual, sides[0], sides[1])),
    Pipeline("eq") >> consume({"DEQ"}) >> Fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), Equal,        sides[0], sides[1])),
    Pipeline("ne") >> consume({"TEQ"}) >> Fwd("parse_comparison") >> set_result(2, lambda sides: correct_right_recursion((Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual), NotEqual,     sides[0], sides[1])),
))
parse_expr = Pipeline("expr") >> parse_comparison  # The expression parser

## Top level construct parsers

parse_variable_suffix_or_call_suffix = Pipeline("varsuffix") >> ambig_lookahead(
    Pipeline("varsuffix-attr") >> ambig_lookahead(
        Pipeline("varsuffix-attr-name") >> consume({"DOT"}) >> next_token({"NAME"}) >> set_result(1, lambda prevs: String(prevs[0][1])),
        Pipeline("varsuffix-attr-expr") >> consume({"LBRACKET"}) >> parse_expr >> consume({"RBRACKET"})
    ) >> set_result(2, lambda prevs: Index(*prevs)) >> optional(Fwd("parse_variable_suffix_or_call_suffix")),
    Pipeline("varsuffix-call") >> parse_call_suffix >> set_result(2, lambda prevs: Call(*prevs)) >> optional(Fwd("parse_variable_suffix_or_call_suffix"))
)
parse_varlist_or_call = Pipeline("varlist") >> ambig_lookahead(
    Pipeline("varlist-parenexpr") >> parse_surrounded(Fwd("parse_expr"), "LPAREN", "RPAREN") >> parse_variable_suffix_or_call_suffix,
    Pipeline("varlist-name") >> parse_name >> optional_lookahead(parse_variable_suffix_or_call_suffix),
) >> ambig(
    Pipeline("varlist-rest") >> consume({"COMMA"}) >> (lambda toks, prevs: ErrorParserState(toks, toks.toks[toks.idx], frozenset()) if isinstance(prevs[-1], Call) else OKParserState(toks, prevs)) >> Fwd("parse_varlist_or_call"),
    set_result(0, lambda _: ())
) >> set_result(2, lambda prevs: prevs[0] if isinstance(prevs[0], Call) else (prevs[0], *prevs[1]))

parse_function_block = Pipeline("func-block") >> Fwd("parse_end_block")

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
        Pipeline("if-end") >> Fwd("parse_end_block") >> set_result(2, lambda prevs: If(prevs[0], prevs[1])),
        Pipeline("if-else-end") >> Fwd("parse_else_block") >> Fwd("parse_end_block") >> set_result(3, lambda prevs: IfElse(prevs[0], prevs[1], prevs[2])),
    ),

    Pipeline("while") >> consume({"WHILE"}) >> parse_expr >> consume({"DO"}) >> Fwd("parse_end_block") >> set_result(2, lambda prevs: While(prevs[0], prevs[1])),
    Pipeline("for") >> consume({"FOR"}) >> next_token({"NAME"}) >> consume({"EQUALS"}) >> parse_expr >> consume({"COMMA"}) >> parse_expr >> ambig_lookahead(
        consume({"COMMA"}) >> parse_expr, set_result(0, lambda _: None)
    ) >> consume({"DO"}) >> Fwd("parse_end_block") >> set_result(5, lambda prevs: For(prevs[0][1], *prevs[1:])),

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

_Pipeline.resolve_fwd()

## End of parser code

grammar = parse_chunk
__all__ = ("grammar",)

def main():
    """test main"""
    import time  # pylint: disable=import-outside-toplevel

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
        parse(grammar, tokens)
        print("done parse", filename)
    print(f"{1000 * (time.perf_counter() - start):.2f} ms")

if __name__ == "__main__":
    main()
