# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

import re
import string

class ParseError(Exception):

    def __init__(self, pos: tuple[str, int, int], err: str):
        self.pos = pos
        self.err = err

    def __str__(self):
        return f"{self.pos[0]}:{self.pos[1]}:{self.pos[2]} {self.err}"

SPECIAL = {
    "function": "FUNCTION", "end": "END", "local": "LOCAL", "return": "RETURN", "if": "IF",
    "then": "THEN", "else": "ELSE", "while": "WHILE", "do": "DO", "elseif": "ELSEIF",
    "for": "FOR", "nil": "NIL", "true": "TRUE", "false": "FALSE", "or": "OR", "and": "AND",
    "not": "NOT", "break": "BREAK",
    "=": "EQUALS", ".": "DOT", ":": "COLON", ";": "SEMICOLON", "(": "LPAREN", ")": "RPAREN",
    "[": "LBRACKET", "]": "RBRACKET", ",": "COMMA", "+": "PLUS", "-": "DASH", "*": "STAR",
    "/": "SLASH", "{": "LCURLY", "}": "RCURLY", "#": "HASH",
    ">": "GT", "<": "LT", "~=": "TEQ", "==": "DEQ", ">=": "GEQ", "<=": "LEQ", "..": "CONCAT"
}
ESCAPE_CHARS = ".()[]++*{}\\|"

RE_WHITESPACE = re.compile(r"[ \t\r\n]")
RE_COMMENT = re.compile(r"--(([^\[\n].*?)?(\n|\Z)|\[\[[\S\s]*?\]\])")
RE_ELLIPSIS = re.compile(r"\.\.\.")
RE_SPECIAL = re.compile("|".join(
    "".join(f"\\{char}" if char in ESCAPE_CHARS else char for char in s) + "(?![A-Za-z0-9_])" * (s[-1] in string.ascii_letters)
    for s in sorted(SPECIAL, key=len)[::-1]
))
RE_NUMBER = re.compile(r"0|[1-9][0-9]*(\.[0-9]*)?")
RE_QSTR = re.compile(r"'([^\n\\']|\\.)*'" + r'|"([^\n\\"]|\\.)*"')
RE_QSTR_SUB = re.compile(r"\\(.)")
RE_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

def lexer(text: str, filename: str):
    if not text.strip():
        yield ((filename, 1, 1), ("EOF", None))
        return

    i = 0
    line = 1
    col = 1
    match = None
    while i < len(text):
        if (match := RE_WHITESPACE.match(text[i:])) or (match := RE_COMMENT.match(text[i:])):
            pass

        elif (match := RE_ELLIPSIS.match(text[i:])):
            yield ((filename, line, col), ("ELLIPSIS", None))

        elif (match := RE_SPECIAL.match(text[i:])):
            yield ((filename, line, col), (SPECIAL[match.group(0)], None))

        elif (match := RE_NUMBER.match(text[i:])):
            yield ((filename, line, col), ("NUMBER", match.group(0)))  # pass as string to avoid conversion-related precision issues

        elif (match := RE_QSTR.match(text[i:])):
            yield ((filename, line, col), ("QSTR", RE_QSTR_SUB.sub(r"\1", match.group(0)[1:-1])))  # TODO: Resolve escapes

        elif (match := RE_NAME.match(text[i:])):
            yield ((filename, line, col), ("NAME", match.group(0)))

        else:
            raise ParseError((filename, line, col), f"unexpected character '{text[i]}'")

        skiptext = match.group(0)
        i += len(skiptext)
        col += len(skiptext)
        line += skiptext.count("\n")
        if (newline_reverse_pos := skiptext[::-1].find("\n")) != -1:
            col = newline_reverse_pos + 1

class Tokens:

    def __init__(self, toks, idx):
        self.toks = toks
        self.idx = idx

    def __hash__(self):
        return hash((self.toks, self.idx))

    @classmethod
    def from_string(cls, text, filename):
        return Tokens(tuple(lexer(text, filename)), 0)

    def advance(self):
        return (self.toks[self.idx] if self.idx < len(self.toks) else (self.toks[-1][0], ("EOF", None)), Tokens(self.toks, self.idx + 1))

class TokenStream:

    def __init__(self, text: str, filename: str, rewind_limit: int = 2):
        self.rewind_limit = rewind_limit
        self.buf = []
        self.i = 0
        self.lasti = 0
        self.lexer = lexer(text, filename)

    @property
    def pos(self):
        return self.buf[self.lasti][0]

    def rewind(self) -> None:
        assert self.i > 0
        self.i -= 1

    def next(self, expect: set[str] | None = None) -> tuple[tuple[str, int, int], tuple[str, str]]:
        while self.i >= len(self.buf):
            try:
                self.buf.append(next(self.lexer))
            except StopIteration:
                self.buf.append((self.buf[-1][0], ("EOF", None)))
            if len(self.buf) > self.rewind_limit:
                self.buf.pop(0)
                self.i -= 1
        res = self.buf[self.i][1]
        if expect is not None and res[0] not in expect:
            expect_list = list(expect)
            if len(expect_list) == 1:
                raise ParseError(self.pos, f"expected {expect_list[0]}, got {res[0]}")
            if len(expect_list) == 2:
                raise ParseError(self.pos, f"expected {expect_list[0]} or {expect_list[1]}, got {res[0]}")
            raise ParseError(self.pos, f"expected {', '.join(expect_list[:-1])}, or {expect_list[-1]}, got {res[0]}")
        self.lasti = self.i
        self.i += 1
        return res
