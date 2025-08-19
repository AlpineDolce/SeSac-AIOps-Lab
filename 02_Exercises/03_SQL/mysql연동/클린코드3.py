nums = (1, 1, 2, 3, 5, 8, 13, 21)
print( nums.__getitem__(0))
print( nums.__getitem__(1))
print( nums.__getitem__(2))

class MyList:
    def __init__(self, data):
        self.data = data
    #연산자 중복
    def __getitem__(self, index):
        print(f"__getitem__ 호출: index = {index}")
        return self.data[index]

mylist = MyList([10, 20, 30])
print(mylist[1])  # -> __getitem__ 호출: index = 1
