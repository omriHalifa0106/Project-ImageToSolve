import Errors


def factorial(a):
    if a - int(a) == 0:
        a = int(a)
        result = 1
        for i in range(1, a + 1):
            result *= i
        return result
    else:
        print(Errors.Undefined_result)
        return None


# Function to perform arithmetic operations.
def apply_operation(a, b, op):
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        try:
            return a / b
        except ZeroDivisionError:
            print(Errors.Zero_division)
            return None
    if op == '@':
        return (a + b) / 2
    if op == '^':
        if b - int(b) != 0 and a < 0:
            a = str(a)
            b = str(b)
            print("Error:", a + "^" + b, "is undefined!")
            return None
        return pow(a, b)
    if op == '%':
        try:
            return a % b
        except ZeroDivisionError:
            print(Errors.Zero_division)
            return None
    if op == '$':
        return (a + b) / 2 + abs(a - b) / 2
    if op == '&':
        return (a + b) / 2 - abs(a - b) / 2
    if op == '~':
        return a * -1
    if op == '!':
        try:
            return float(factorial(a))
        except TypeError:
            return None


def apply_operation2(a, op):
    return apply_operation(a, a, op)
