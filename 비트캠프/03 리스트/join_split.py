'''
배열 :  문자열, 리스트, 튜플, 딕셔너리, 셋...
A.join(B) : 결과값은 "문자열"
배열.split(값) : 결과값은 [리스티]
'''
'''
my_list = ['Life', 'is', 'too', 'short']
# Life is too short 출력
var = " ".join(my_list)
print(var)
'''
my_str = "beetes"
# BTS 출력

'''
var2 = my_str.replace("e","")

print(var2.upper())
'''
var3 = my_str.split("e") # ['b', '', 't', 's']
var3 = "".join(var3)

print(var3.upper())
