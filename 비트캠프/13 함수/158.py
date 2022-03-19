'''
def add(a, b):
    print( a, "과" ,b , "의 합은" ,a + b , "입니다.")

add(3, 4)
'''
'''
def add_many(*a): #매개변수:튜플
    return sum(a)



print(add_many(3, 4, 5, 6 ,7))
'''
def add_mul(a, *b):
    print(a, b)
    if a == 'add':
        return sum(b)
    elif a == 'mul':
        gop = 1
        for i in b:
            gop *= i
        return gop
    else:
        return -1

print(add_mul('add',1, 2, 3, 4, 5))
print(add_mul('muo',1, 2, 3, 4, 5))
