function f(a)
    return function()
        print(a)
    end
end

local g = f(10)
g()
