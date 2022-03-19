#리스트.append(값) - 리스트의 마지막 위치
a = [1, 2, 3]
a.append(4)
#a.append(5, 6) #에러, 하나씩만
a.append([5, 6]) #다중리스트
print(a)

# 리스트.insert(위치, 값)
a = [1, 2, 3]
a.insert(3, 5)
print(a)


# 리스트.extend([리스트])
a = [1, 2, 3]
a.extend([4, 5])
print(a)
