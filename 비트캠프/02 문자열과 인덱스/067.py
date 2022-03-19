# 문자열 함수

a = "hobby"
print(a.count('b')) #a 라는 문자에 b 라는 문자 갯수

a = "Life is the best choice"
print(a.find('b')) # 12, 찾는 문자의 위치(인덱스)
# 인덱싱 : 배열(문자열)의 방번호, 대괄호[]이용, 0~ 시작
print(a.find('i')) #1, 중복문자는 좌측부터 최초위치
print(a.find('i', 10)) # 20, 특정 인덱스 이후부터 검색
# 찾는문자가 없으면 결과값 -1

a = "Life is too short"
print(a.index('t')) # 8, find()와 동일
#찾는 문자가 없으면 오류!  find()와 다름

print(",".join('abcd')) # A.join(B) 'A문자가 끼어든다 B에'

a = " Hi "
print(a. upper()) #대문자
print(a. lower()) #소문자
print(a. swapcase()) #대소문자 반전
print(a. lstrip()) # 좌측 공백제거
print(a. rstrip())
print(a. strip()) #양쪽 공백제거

a = "Life is too short"
print(a.replace("Life", "Your leg")) # 문자열.replace(A, B)
print(a) #원본은 보전

print(a.split()) #문자열.split() : 문자열을 공백으로 나누어 [리스트]로 반환
b = "a : b : c : d"
print(b.split(':'))


