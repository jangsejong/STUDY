#피보나치 수열 구하기
# 1, 2, 3, 5, 8, 13 ,21 ...

n = int(input("얼마까지 피보나치 수열을 구할깡?:"))

p = [1, 2]

while True:
    i = p[-2] + p[-1]
    if i > n:
        break
    p.append(i)

print(p)
