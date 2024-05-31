-- inline unpack implementation in case the `table` module is not present
local function unpack(t, i, j)
    if i > j then
        return
    end
    return t[i], unpack(t, i+1, j)
end

function assert(v, message)
    if bor(v == nil, v == false) then
        error(bor(message, "assertion failed!"))
    end
end

function collectgarbage(opt, arg)
    error("collectgarbage not implemented")
end

function dofile(filename)
    error("dofile not implemented")
end

function getmetatable(obj)
    error("getmetatable (and metatables in general) not implemented")
end

function iterfor(iterable, body)
    -- Temporary replacement for the for ... in ... do ... end construct
    local iterator
    local invariant
    local state
    iterator, invariant, state = iterable()
    while true do
        local values = {iterator(invariant, state)}
        if values[1] == nil then
            break
        end
        state = values[1]
        body(unpack(values))
    end
end

function ipairs(t)
    local i = 1
    local j = #t
    return function()
        if i > j then
            return nil
        end
        i = i + 1
        return t[i - 1]
    end
end

function loadfile(filename, mode, env)
    error("loadfile not implemented")
end

function pairs(t)
    return next, t, nil
end

function pcall(func, ...)
    -- The interpreter detects frames that have a local called _ERR and doesn't propagate errors across them, putting the exception into the local instead
    local _ERR
    local results = {func(...)}
    if _ERR == nil then
        return true, unpack(results, 1, #results)
    end
    return false, _ERR
end

function rawequal(v1, v2)
    error("rawequal not implemented")
end

function rawget(table, index)
    error("rawget not implemented")
end

function rawlen(v)
    error("rawlen not implemented")
end

function rawset(t, i, v)
    error("rawset not implemented")
end

function select(index, ...)
    if index == "#" then return #{...} end
    assert(type(index) == "number", "select: parameter #1 must be number or \"#\"")
    local args = {...}
    if index < 0 then index = #args + index + 1 end
    local result = {}
    for i=index,#args do
        result[i-index+1] = args[i]
    end
    return unpack(result)
end

function setmetatable(t, mt)
    error("setmetatable (and metatables in general) not implemented")
end

_VERSION = "Lua 5.4"

function xpcall(f, msgh, ...)
    error("xpcall not implemented")
end
