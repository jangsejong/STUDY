#피보나치 수열 구하기
# 1, 2, 3, 5, 8, 13 ,21 ...
'''
n = int(input("얼마까지 피보나치 수열을 구할깡?:"))

p = [1, 2]

while True:
    i = p[-2] + p[-1]
    if i > n:
        break
    p.append(i)

print(p)
'''
#사용자 정의 함수
'''
def add(a, b):
    print(a + b)
    
add(10, 20)

'''
'''
a = int(input("더할 첫째 숫자 입력하세요:"))
b = int(input("더할 두번째 숫자 입력하세요:"))

def add(a , b): # a, b: 매개변수,argument
    print(a + b)

c = add(a, b) # 10, 20: 인수, parameter
'''
def say(str):
    return str


print(say("ho")) #ho 출력 
5
