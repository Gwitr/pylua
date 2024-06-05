table = {}

function table.insert(t, pos_or_item, item_or_nil)
    if type(t) ~= "table" then
        error("table.insert parameter #1 must be table, got " .. type(t))
    end
    if item_or_nil == nil then
        t[1+#t] = pos_or_item
        return
    end
    for i=#t,pos_or_item,-1 do
        t[i+1] = t[i]
    end
    t[pos_or_item] = item_or_nil
end

function table.concat(t, sep, i, j)
    sep = bor(sep, "")
    i = bor(i, 1)
    j = bor(j, #t)
    local result = ""
    local first = true
    for idx=i,j do
        if first then
            -- TODO: not opcode
        else
            result = result .. sep
        end
        result = result .. t[idx]
        first = false
    end
    return result
end

function table.move(a1, f, e, t, a2)
    if a2 == nil then
        a2 = a1
    end
    local temp = {}
    for i=f,e do
        temp[i-f+t] = a1[i]
    end
    for i=t,e-f+t do
        a2[i] = temp[i]
    end
    return a2
end

function table.pack(...)
    -- TODO: use pairs(...) to figure out the true number of arguments. This will fail on nil!!
    return { n=#{...}, ... }
end

function table.remove(t, pos)
    pos = bor(pos, #t)
    for i=pos+1,#t do
        t[i-1] = t[i]
    end
    t[#t] = nil
end

function table.sort(t, comp)
    error("Not implemented")
end

function table.unpack(t, i, j)
    i = bor(i, 1)
    j = bor(j, #t)
    if i > j then
        return
    end
    return t[i], table.unpack(t, i+1, j)
end

local function dump_table(t)
    print("size", #t)
    for i=1,#t do
        print(i, t[i])
    end
end
