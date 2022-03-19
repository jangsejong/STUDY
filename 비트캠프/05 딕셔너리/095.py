#딕셔너리 {1:'a', 2:'b', 3:'c'
# {키1:값, ...} , 키는 유니크하고 읽기전용
'''
a = {'이름':'홍길동', '나이': 23 , '주소':'서울' }
print(a['이름'])

a['전화'] = '010'  #추가
print(a)

a[1] =100
print(a)

#a[[4]] =123 #키는 읽기전용 

a[(4,)] =123
print(a)

a[1] = 200 # 키 1이 이미 존재하므로 추가가 아닌 수정임

del a[1]
print(a)
'''
'''
a = {'이름':'홍길동', '나이': 23}

print(a.keys())
print(list(a.keys()))

print(a.values())
print(list(a.values()))

print(a.items())
print(list(a.items()))
'''
a = {'이름':'홍길동', '나이': 23 , '주소':'서울' }
del a['나이']
print(a)

#a.clear()
#a= {}
print(a['이름'])
print(a.get('이름')) #상동

#print(a['나이']) #키가 없으면 에러
print(a.get('나이'))  #키가 없으면 none
print(a.get('나이',20))

print('주소' in a ) #a 안에 주소 있니

