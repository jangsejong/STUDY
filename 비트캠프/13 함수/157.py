'''
def add(a, b):
    print( a, "과" ,b , "의 합은" ,a + b , "입니다.")

add(3, 4)
'''
def add_many(*a): #매개변수:튜플
    return sum(a)



print(add_many(3, 4, 5, 6 ,7))
