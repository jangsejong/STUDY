m = int(input("월을 입력하세요:"))
#A)
'''
if m == 1 :
    print("31")
elif m== 2 :
    print("28")
elif m == 3 :
    print("31")
elif m == 4 :
    print("30")
elif m == 5 :
    print("31")
elif m == 6 :
    print("30")
elif m == 7 :
    print("31")
elif m == 8 :
    print("31")
elif m == 9 :
    print("30")
elif m == 10 :
    print("31")
elif m == 11 :
    print("30")
else :
    print("31")
'''
#B)
'''
if m == 2 :
    print("28")

elif m == 4 or m == 6 or m == 9 or m == 11 :
    print("30")
else :
    print("31")
'''
    
#C)
'''
if m == 2 :
    print("28")
elif m in [4, 6, 9, 11] :
    print("30")
else :
    print("31")
'''
#D)
'''
if m == 2 :
    print("28")
elif m in [4, 6, 9, 11] :
    print("30")
elif m in [1, 3, 5, 7, 8, 10, 12] :
    print("31")    
else :
    print("1~12 사이만 입력하세요!")
'''
#E)
'''
if m < 1 or m > 12 :
    print(" 1~12 사이만 입력하세요!")
else:
    if m == 2 :
        print("28")
    elif m in [4, 6, 9, 11] :
        print("30")
    else :
        print("31")
'''

#11월 은 30일 까지입니다
if m < 1 or m > 12 :
    print(" 1~12 사이만 입력하세요!")
    quit()
else:
    if m == 2 :
        day = 28
    elif m in [4, 6, 9, 11] :
        day = 30
    else :
        day = 31
    print("%d 월은 %d 일 까지입니다" % (m , day))
