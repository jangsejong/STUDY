hap = 0

#1~100까지의 3의 배수 합은?
'''
for i in range(3,101,3):
    hap += i
'''
'''
for i in range(1,101):
    if i % 3 == 0 :
        hap += i

print(hap)
'''
# 1~100 까지 3배수를 제외한 합은?
hap = 0
'''
for i in range(1,101):
    if i % 3 != 0:
        hap += i 


print(hap)
'''
hap = 0
for i in range(1,101):
    if i % 3 != 0:
        continue #인접 순환을 탈출하여 위로 이동
    hap += i


print(hap)
