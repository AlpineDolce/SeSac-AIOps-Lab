def some_func():
    """Hello Function Docstring"""
    print("함수에 설명서를 달아보자")

def add(x, y):
    """
    이 함수는 두개의 정수를 받아가서 더하여 그 결과를 반환합니다. 
    """
    return x+y 


print( print.__doc__)
print( some_func.__doc__)
print( add.__doc__)

class Score:
    """
    이 클래스는 성적처리를 담당하는 클래스입니다
    """
    name=""

print( Score.__doc__)


class User:
    def __init__(self, name="", age=0):
        self.user_name = name 
        self.user_age = age 

user = User("홍길동", 23) 

def get_user_name(user):
    """
        Description: 
            유저의 객체를 통해 유저의 이름을 알아내는 함수.
        Param:
            User 클래스의 객체를 파라미터로 받음.
        Return:
            User 클래스의 userName 객체를 반환.     
    """

    return user.user_name

print(get_user_name.__doc__)
print( get_user_name(user))


def string_reverse(str1):
    """
    Returns the reversed String.

    Parameters:
       str1 (str):The string which is to be reversed.

    Returns:
       reverse(str1):The string which gets reversed.   
    """
    reverse_str1 = ''
    i = len(str1)
    while i > 0:
        reverse_str1 += str1[i - 1]
        i = i- 1
    return reverse_str1

print(string_reverse.__doc__)
print(string_reverse("korea"))


def square(a):
    """Returns the square of the given number."""
    return a ** 2  # Corrected exponentiation

# Accessing the docstring
print(square.__doc__)