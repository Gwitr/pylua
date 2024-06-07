-- inline unpack implementation in case the `table` module is not present
local function unpack(t, i, j)
    if i > j then
        return
    end
    return t[i], unpack(t, i+1, j)
end

-- The coroutine table is created in llib.py

function coroutine.isyieldable()
    return true
end

function coroutine.wrap(f)
    local coro = coroutine.create(f)
    return function(...)
        local res = {coroutine.resume(coro, ...)}
        if res[1] then
            return unpack(res, 2, #res)
        end
        error(res[2])
    end
end
