#직사각형 만들기
'''
*****
*****
*****
*****
*****
'''
'''
for i in range(1, 6): #줄
    for j in range(1,6): #별
        print("*", end="")
    print() #줄바꿈 

#직삼각형만들기
'''
'''
*
**
***
****
*****
'''
'''
for i in range(1,6):
    for j in range(1, i+1):
        print("*", end="")
    print()
'''
'''
for i in range(1,6):
    print("*"* i)
'''
#다른 직삼각형 만들기
'''
for i in range(1,6):
    print(" "*(5-i)+"*"*i, end="")
    print()
'''
'''
for i in range(1,6):
    a= "*"*i
    print('%7a' % a)
'''
'''
A = [70, 65, 55, 75, 95, 80, 85, 100]

#print(sum(A)/len(A))
count = 0
total = 0
for i in A:
    if i >=80 :
        count += 1
        total += i

print(total/count)
'''
#사용자가 입력한 문자열에서 숫자를 제거

a = input("문자열을 입력하세요 : ")
b = ""

#A)
for i in range(0, len(a)):
    if a[i].isdigit() == False: #문자라면
        b += a[i]
        #b.append(a[i]) 문자열은 수정/삭제안됨

print(b)

#B)

for i in a:
    it not i.isdogit():
        b +=i
print(b)
