#리스트.sort()- 오름차순
a = [1, 4, 3, 2]
a.sort()
print(a)

#리스트.sort(reverse=True)- 내림차순
a = [1, 4, 3, 2]
a.sort(reverse=True)
print(a)

#reverse() - 좌우반전 
a.reverse()
print(a)

a = [3, 2, 1, 4, 6, 5, 7, 9, 8, 0 ]
'''
미션: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
'''

#a.sort(reverse=True) - 아래와 결과값 동일

a.sort()
a.reverse()
print(a)
