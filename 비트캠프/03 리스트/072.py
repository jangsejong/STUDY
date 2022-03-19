score = [90, 87, 75, 86.5, "장세종"] #리스트

print(score[0]) #90 , 인덱스
print(score[4:]) #장세종

j = ['a', 'b', 'c']
print(j) 
j[0]= 'd' #리스트 수정 가능
print(j) #[d, b, c]

i = "abc"
print(i)
#i[0] = 'd' #문자열은 수정불가


#다중리스트
a = [1, 2, 3, ['a','b','c']]
print(a[-1])
print(a[-1][0]) #2차원리스트 ,출력값 a

