
#A)
a = 1
def vartest(a): #함수 외부의 a와는 별개(동명이인, 함수내에서만 통용) 
        a += 1
        print("함수내에서 a는 %d 입니다" % a)
vartest(a)
print(a) #1


#B)
a = 1
def vartest2(b): 
        b += 1
        print("함수내에서 b는 %d 입니다" % b)
vartest2(a)
print(a) #1


#C)
a = 1
def vartest3():
        global a #함수내부에서 외부의 변수를 직접 사용하겠다는 의미 
        a += 1
        print("함수내에서 a는 %d 입니다" % a)
vartest3()
print(a) #1
