# Set : 집합 {}, 딕셔너리와 유사(단, 키만 나열) , 중복제거 

s1 = {7, 1, 1, 2, 3}
print(type(s1))
print(s1)

#s2 = {} #딕셔너리
s2 = set()

s3 = "hello"
s3 = set(s3)
print(s3)

s4 = {1, 2, 3}
#print(s4[0]) 에러

s5 =list(s4)
print(s5)
print(s5[0]) 
