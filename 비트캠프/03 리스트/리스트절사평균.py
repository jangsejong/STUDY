

a = [5, 5, 1, 3, 4, 2, 1]
'''
미션 :  리스트의 절사평균을 구하시오.
- 절사평균이란 최소값과 최대값을 제외한 나머지값들의 평균
-리스트의 합: sum(리스트명)  예)print(sum(a))
- 단, 어떠한 리스트가 주어져도 수행될 수 있어야 함

'''
a = [5, 5, 1, 3, 4, 2, 1]
a.sort()
del a[0]
a.pop()
print(sum(a)/len(a))

#B)
print(min(a))
print(max(a))
a.remove(min(a))
a.remove(max(a))
print(sum(a)/len(a))



