# (tuple) -  읽기전용, 소괄호 생략 가능 
a = (1, 2, 3)
print(type(a))
#a[0] = 4   # 에러
#del a[0] # 에러

a = 1, 2, 3
a = ()
a = (1,)
a = 1,
print(type(a))
print(a)

b = list(a)
print(b)

a = tuple(b)
print(a)

print(a[0]) # 리스트 또는 인덱스
