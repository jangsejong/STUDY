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
for i in range(1,6):
    a= "*"*i
    print('%7a' % a)
