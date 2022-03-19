#실전 로또

import random

a = [] #채울 배열
b = [17, 18, 21, 27, 29, 33] #당첨번호
money = 0 #투기금액 

def lotto():
    while len(a) < 6: #6개 미만
        r = random.randrange(1, 46)
        if a.count(r) == 0: # 새값
            a.append(r) #배열확정
    a.sort() #오름차순

while a != b: #꽝이라면
    a.clear() #배열지움
    lotto() #다시뽑기
    money += 1000 # 금액증가
    for i in a:
        print("%2d" % i, end=" ")
    print("\t\t" + format(money, ',')+"원") #금액장식 

print("이제 나도 건물주!")

'''
for i in range(1, 7):
    r = random.randrange(1, 46)
    print(r)
'''
