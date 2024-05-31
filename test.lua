print("Hello, world!")

for i=1, 10, 2 do
    print(i)
end

local t = {1, 2, 3}
table.insert(t, 10)
table.insert(t, 20)
for i=1,#t do
    print(i, t[i])
end

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
