# 미션
# 두 정수를 사용자로부터 입력받아서
#사칙연산의 결과를 출력
# 출력 예)
# 10 + 20 = 30
# 10 - 20 = 30
# 10 * 20 = 30
# 10 / 20 = 30

a = int(input("첫째 정수를 입력해 주세요 : "))
b = int(input("둘째 정수를 입력해 주세요 : "))
print ( "%d + %d = %d"% (a, b, a+b))
print( "a - b = ", a - b )
print( "a * b = ", a * b )
print( "a / b = ", a / b )
print("%d / %d 의 몫은 %d"% (a , b, a//b )) #// 몫, %나머지, **제곱
print("%d 의 %d 제곱은 %d"% (a , b, pow(a, b )))
