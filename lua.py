"""The Lua command line tool"""

import llib
import ltypes
import linterp

def main():
    """Entry point for the Lua command line tool"""
    interp = linterp.Interpreter(ltypes.Table())
    llib.load(interp)
    def entry():
        with open("test.lua", encoding="ascii") as f:
            func, *rest = yield from interp.call(interp.global_env[ltypes.String(b"load")], [f.read()])
        if func is ltypes.nil:
            raise ltypes.InterpreterError("syntax error: " + rest[0].s.decode("utf8"))
        return (yield from interp.call(func, []))
    interp.mainloop(entry())

if __name__ == "__main__":
    main()
