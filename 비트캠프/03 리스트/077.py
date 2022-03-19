a = [1, 2, 3]
a[1] = 4
print(a)

#리스트.remove (값)
b = [1, 2, 3, 1, 2, 3]
b.remove(3) # 최초로 나온 값을 삭제
print(b)

#리스트.pop() 맨 마지막요소를 반환

a = [1, 2, 3]
print(a.pop())
print(a)

#리스트.pop (위치)
a = [1, 2, 3]
print(a.pop(1))
print(a)

b = "xyz"
#print(b.pop(0)) 문자열.pop()은 사용불가

