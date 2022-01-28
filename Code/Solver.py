import Calculation_module
import Errors
from Precedence import precedence


def goto(linenum):
    """
    :param linenum: the number of situation we in
    :return: the function returns linenum
    """
    return linenum


def right_left(exp):
    """
    :param exp: : the string for evaluation
    :return: return true if the expression is valid
    """
    for i in range(0, len(exp)):
        if exp[i] == "!":
            if i != len(exp) - 1 and exp[i + 1].isdigit():
                return False
        if exp[i] == "~":
            if i != 0 and exp[i - 1].isdigit():
                return False
    return True


def sign(exp):
    """
    :param exp: the string for evaluation
    :return: the string with minus as sight replaced with tilda
    """
    exp = exp.replace("(-", "(~")
    exp = exp.replace("^-", "^~")
    exp = exp.replace("*-", "*~")
    exp = exp.replace("--", "-~")
    exp = exp.replace("+-", "+~")
    exp = exp.replace("/-", "/~")
    exp = exp.replace("%-", "%~")
    exp = exp.replace("$-", "$~")
    exp = exp.replace("&-", "&~")
    exp = exp.replace("@-", "@~")
    exp = exp.replace("!-", "!~")
    exp = exp.replace("~-", "~~")
    return exp


def is_valid(expression_string):
    """
    :param expression_string: The expression we would like to solve as a string.
    :return: the function returns whether or not the string is contains invalid chars
    """
    if all(ch.isdigit() or ch == "." or ch == "(" or ch == ")" or ch == " " or ch == "\t" or
           ch in precedence for ch in expression_string):
        return True
    return False


def is_empty(expression_string):
    """
    :param expression_string: The expression we would like to solve as a string.
    :return: the function returns whether or not the string is contains invalid chars
    """
    if expression_string is "":
        return True
    return False


def built_num(expression_string, i):
    """
    :param expression_string: The expression we would like to solve as a string.
    :param i: the position in the string
    :return: the function returns a number as a float from the string
    """
    num = ""
    # There may be more than one
    # digits in the number.
    while i < len(expression_string) and (expression_string[i].isdigit() or expression_string[i] == '.'):
        num += expression_string[i]
        i += 1
    if num[0] == '.' or num[len(num) - 1] == '.':
        print(Errors.Illegal_number)
        return None
    try:
        num = float(num)
        return num, i
    except ValueError:
        print(Errors.Illegal_number)
        return None


def is_one_operand(operator):
    """
    :param operator: an operator
    :return: the function returns whether or not the operator gets one operand
    """
    if operator == '!' or operator == '~':
        return True
    return False


def evaluate_expression(expression_string):
    """
    :param expression_string: The expression we would like to solve as a string.
    :return: The result of said evaluation.
    """
    line = 1
    if is_valid(expression_string):
        expression_string = expression_string.replace(" ", "")
        expression_string = expression_string.replace("\t", "")
        if "~~" in expression_string:
            print(Errors.Illegal_expression)
            return None
        if is_empty(expression_string):
            print(Errors.EmptyString)
            return None
        if not right_left(expression_string):
            print(Errors.Illegal_expression)
            return None
        operands = []
        # stack to store operators.
        operators = []
        i = 0
        expression_string = sign(expression_string)
        while i < len(expression_string):
            # Current token is an opening
            # brace, push it to 'ops'
            if expression_string[i] == '(':
                operators.append(expression_string[i])

            # Current token is a number, push
            # it to stack for numbers.
            elif expression_string[i].isdigit() or expression_string[i] == '.':
                num = ""
                # There may be more than one
                # digits in the number.
                while i < len(expression_string) and (expression_string[i].isdigit() or expression_string[i] == '.'):
                    num += expression_string[i]
                    i += 1
                if num[0] == '.' or num[len(num) - 1] == '.':
                    print("Error: The value is not number!")
                    return None
                try:
                    num = float(num)
                except ValueError:
                    print("Error: The value is not number!")
                    return None

                operands.append(num)

                # right now the i points to
                # the character next to the digit,
                # since the for loop also increases
                # the i, we would skip one
                # token position; we need to
                # decrease the value of i by 1 to
                # correct the offset.
                i -= 1

            # Closing brace encountered,
            # solve entire brace.
            elif expression_string[i] == ')':

                while len(operators) != 0 and operators[-1] != '(':

                    if not is_one_operand(operators[-1]):
                        try:
                            val2 = operands.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            val1 = operands.pop()
                        except IndexError:
                            if operators[-1] == '-':
                                operators[-1] = '~'
                                op = operators.pop()
                                result = Calculation_module.apply_operation2(val2, op)
                                if result is not None:
                                    operands.append(result)
                                    continue
                                else:
                                    return None
                            else:
                                print(Errors.Illegal_expression)
                        try:
                            op = operators.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            operands.append(Calculation_module.apply_operation(val1, val2, op))
                        except OverflowError:
                            print(Errors.OverFlow)
                            return None
                    else:
                        try:
                            val1 = operands.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            op = operators.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            operands.append(Calculation_module.apply_operation2(val1, op))
                        except OverflowError:
                            print(Errors.OverFlow)
                            return None

                # pop opening brace.
                operators.pop()

            # Current token is an operator.
            else:

                # While top of 'ops' has same or
                # greater precedence to current
                # token, which is an operator.
                # Apply operator on top of 'ops'
                # to top two elements in values stack.
                while len(operators) != 0 and precedence.get(operators[-1]) >= precedence.get(expression_string[i]):
                    if expression_string[i] == '-' and expression_string[i - 1] in precedence:
                        operators.append(expression_string[i])
                        operators[-1] = '~'
                        line = goto(2)
                        break
                    if expression_string[i] == '~':
                        break
                    if not is_one_operand(operators[-1]):
                        try:
                            val2 = operands.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            val1 = operands.pop()
                        except IndexError:
                            if operators[-1] == '-':
                                operators[-1] = '~'
                                op = operators.pop()
                                result = Calculation_module.apply_operation2(val2, op)
                                if result is not None:
                                    operands.append(result)
                                    continue
                                else:
                                    return None
                            else:
                                print(Errors.Illegal_expression)
                                return None
                        try:
                            op = operators.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            operands.append(Calculation_module.apply_operation(val1, val2, op))
                        except OverflowError:
                            print(Errors.OverFlow)
                            return None
                    else:
                        try:
                            val1 = operands.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            op = operators.pop()
                        except IndexError:
                            print(Errors.Illegal_expression)
                            return None
                        try:
                            operands.append(Calculation_module.apply_operation2(val1, op))
                        except OverflowError:
                            print(Errors.OverFlow)
                            return None

                # Push current token to 'ops'.
                if expression_string[i] in precedence and line != 2:
                    operators.append(expression_string[i])
            line = goto(1)
            i += 1

        # Entire expression has been parsed
        # at this point, apply remaining ops
        # to remaining values.
        while len(operators) != 0:
            if not is_one_operand(operators[-1]):
                try:
                    val2 = operands.pop()
                except IndexError:
                    print(Errors.Illegal_expression)
                    return None
                try:
                    val1 = operands.pop()
                except IndexError:
                    if operators[-1] == '-':
                        operators[-1] = '~'
                        op = operators.pop()
                        try:
                            result = Calculation_module.apply_operation2(val2, op)
                        except OverflowError:
                            print(Errors.OverFlow)
                            return None
                        if result is not None:
                            operands.append(result)
                            continue
                        else:
                            return None
                    else:
                        print(Errors.Illegal_expression)
                        return None

                op = operators.pop()
                try:
                    result = Calculation_module.apply_operation(val1, val2, op)
                except OverflowError:
                    print(Errors.OverFlow)
                    return None
            else:
                try:
                    val1 = operands.pop()
                except IndexError:
                    print(Errors.Illegal_expression)
                    return None
                op = operators.pop()
                try:
                    result = Calculation_module.apply_operation2(val1, op)
                except OverflowError:
                    print(Errors.OverFlow)
                    return None
            if result is not None:
                operands.append(result)
            else:
                print(Errors.Illegal_expression)
                return None
        if len(operands) == 1:
            return operands[0]
        else:
            print(Errors.Illegal_expression)
            return None
    else:
        print(Errors.Invalid_char)
        return None



