#소수 Prime Number : 2, 3, 5, 7, 11, 13, 17, 19 ...

def prime(n):
        for i in range(2, n): #1과 자신의 수를 제외한 범위
                if n % i == 0:
                        print(i) # i 를 찾고 싶을때 
                        return "소수 아닙니다"
        return "소수 입니다"



print(prime(11)) # 소수입니다 출력
print(prime(121)) # 소수가 아닙니다 출력
print(prime(1234567))
