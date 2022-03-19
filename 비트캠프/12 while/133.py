#사용자가 q를 입력하면 그만하기
#A
'''
a = input("q를 입력하면 종료 : ")
while a != 'q':
    print(a)
    a = input("q를 입력하면 종료 : ")
'''

#B
while True:
    a = input("q를 입력하면 종료 : ")
    if a == 'q':
        break
    print(a)

print("종료합니다")
