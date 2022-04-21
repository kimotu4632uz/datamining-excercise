mylist = [3,7,-4,8,435.0,-173.0]

print("配列mylistは {0}".format(mylist))

mylist.append(10.0)
print("配列mylistを更新 -> {0}".format(mylist))
print("１番目は {0}, ４番目は{1}".format(mylist[0],mylist[3]))
print("1番目〜3番目は{0}".format(mylist[:3]))

print("mylistから1つづつ値を取り出して1を足すと")
for i, value in enumerate(mylist):
    value = value * (i + 1)
    print(value)

print("mylistを順番に並べると")
for i, value in enumerate(mylist):
    print("{0} 番目は {1}".format(i+1,value))
    
mylist[1] = 0.001
print("mylistの2番目を書き換えると")
print(mylist)
