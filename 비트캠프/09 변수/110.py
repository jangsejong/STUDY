#철학자 강신주 

a, b = 'python', 'life'
print(a, b)
print(a + b)

a = b = 123
print(a, b)

# 고전적인 값 교체

a = 10
b = 20

temp = a
a = b
b = temp
print(a, b)

# 파이썬 값 교체
a = 10
b = 20
a, b = b, a
print(a, b)
