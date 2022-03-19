#A)
'''
for i in [1, 2, 3, 4, 5, 6 ,7 ,8 ,9, 10]:
    print(i)


#B)
#print(list(range(1,101, 2)))

for i in range(10, 101,10) :
    print(i)

#C) 카운트다운
#for i in range(10, 0, -1) :
#    print(i)

for i in reversed(range(1,11)):
    print(i)
print("발사")
'''
hap = 0
p= int(input( "얼마까지 더할까요?"))
#print(p)
for i in range(1, p+1):
    hap = hap + i #hap += i
print(hap)
