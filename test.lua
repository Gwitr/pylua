print("Hello, world!")

for i=1, 10, 2 do
    print(i)
end

local t = {1, 2, x=2463, 3}
table.insert(t, 10)
table.insert(t, 20)
for i=1,#t do
    print(i, t[i])
end
local key, value
key, value = next(t, key)
print(key, value)
key, value = next(t, key)
print(key, value)
key, value = next(t, key)
print(key, value)
key, value = next(t, key)
print(key, value)
key, value = next(t, key)
print(key, value)
key, value = next(t, key)
print(key, value)
key, value = next(t, key)
print(key, value)

local resultCode
local err
resultCode, err = pcall(function()
    print("This throws an error:")
    error("hello")
end)
if resultCode then
    print("No error was thrown")
else
    print("An error was thrown:", err)
end

-- Coroutines test
local function coro_func(a)
    print("[my_coro] Hi! Arg is", a)
    print("[my_coro]", coroutine.running())
    local x = coroutine.yield(1)
    print("[my_coro] Got", x)
    print("[my_coro]", coroutine.running())
    return "test", "foo", "bar"
end

local coro = coroutine.create(coro_func)
print("[main] Hiii")
print("[main]", coroutine.running())
local cres, x = coroutine.resume(coro, 10)
print("Coroutine is", coroutine.status(coro))
print("[main] :3 I got", x)
print("[main]", coroutine.running())
print("Coroutine is", coroutine.status(coro))
cres, x = coroutine.resume(coro, "wgdhd")
print("Coroutine is", coroutine.status(coro), "and gave", x)
print("[main] Okay the next resume call should fail and return false")
cres, x = coroutine.resume(coro)
if cres then
    print("Inexplicably, it did not")
else
    print("it has indeed done that, and also returned: " .. x)
end
print("Coroutine is", coroutine.status(coro))
