from MathModel import imageToExcersise_FromModel
from Solver import evaluate_expression
from Precedence import precedence
import Calculation_module
import Errors


def main():
    image_path = input("Please enter the exercise image path:")
    str_exercise = imageToExcersise_FromModel(image_path)
    try:
        x = evaluate_expression(str_exercise)
        if x is not None:
            print("Result:", x)
    except KeyboardInterrupt:
        print(Errors.Click)

    except EOFError:
        print(Errors.Invalid_char)

if __name__ == "__main__":
    main()

