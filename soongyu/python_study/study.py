# 파이썬 기본 문법
# 깃허브 사용 방식 연습 및 리마인드용
# 공부하며 필요시 추가 예정

# 변수 선언
a = 'hello world'

# 출력 print
print(a)
# hello world가 출력된다.

# 입력 input
# 사용자의 입력을 받아 "문자열" 형태로 반환한다.
# 입력 받기 위한 문자열 출력이 가능하다.
b = input('입력: ')
print(b)
# 입력 받은 값이 출력된다.

# 조건문 if, elif, else
# 사용법은 C, C++ 등 다른 언어와 유사하다.
number = float(input("숫자를 입력: "))

if number > 0:
    print("양수.")
elif number < 0:
    print("음수.")
else:
    print("0.")
# 입력 값에 따라 결과가 나온다.

# 반복문 for, while
# for문에서 주로 in, of를 사용한다. range를 이용해 범위를 지정한다.
# 사용법은 C, C++ 등 다른 언어와 유사하다.
# for문
my_list = [1, 2, 3, 4, 5]
for item in my_list:
    print(item)
print()
# for문 range 사용
for i in range(1, 11):
    print(i)
print()
# while문
i = 1
while i <= 5:
    print(i)
    i += 1
print()