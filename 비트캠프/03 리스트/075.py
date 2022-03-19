a = [1, 2, 3, 4, 5]
print(type(a))
print(a[0:3])
print(a[:3])
print(a[2:])

a = [1, 2, 3]
b = [4, 5, 6]
print(a + b) # 리스트로 합함
print(a * 3) # 리스트의 반복
 #print(a - b)
 #print(a / b)

a = "abc"
print(len(a)) #3

a = [1, 2, 3]
print(len(a)) #3 길이
print(type(a[2]))
print(str(a[-1]) + 'hi')

a[0] = 4
print(a) #[4, 2, 3]
del b[2:]
print(b)

del b # 리스트 자체를 삭제
#print(b)

c = [1, 2, 3]
del c[:] #리스트의 내용만 삭제, 빈리스트로 남음
print(c)
