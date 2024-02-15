# wap to make a salary prediction system
import numpy as np
from salary_prediction_system.model import *


# define a method to predict a salary based on an user experience
def predicted_salary(user_experience):
    csv_file='attachment_Salary-Data.csv'
    dataframe = read_csv_to_dataframe(csv_file)
    X_train, X_test, y_train, y_test = split_data(dataframe, test_size=0.2, random_state=42)
    model=create_and_train_model(X_train, y_train)
    predicted_data=predict_test_data(model,user_experience)
    return predicted_data


# define a method to clear an output screen
def clear_screen():
    print('\033[H\033[J')


# define a method to show a rules of our system
def about_system():
    print("\nWelcome To This Salary Prediction System : ")
    print("Here user can predict their salary on the basis of their experience.....")
    input("\npress enter to continue....")


# define a method to print a welcome message
def welcome_message():
    user_name = input("Please enter your name: ")
    message = f"Hello, {user_name} ! Welcome to our program."
    print(message)
    print("\nHere enter your experience and predict your salary.....")
    input("\npress enter to continue....")


# main function
if __name__ == "__main__":
    about_system()
    clear_screen()
    welcome_message()
    clear_screen()

    while 1:
        print("\n1. for continue\t2.for exit.")
        ch = int(input("enter your choice : "))

        match ch:
            case 1:
                ex=[]
                user_experience = eval(input("\nenter your experience : "))
                ex.append([user_experience])
                experience =np.array(ex)
                user_salary = predicted_salary(experience)
                print("\nyour salary is %2.2f RS. : "%(user_salary[0]))
            case 2:
                exit()
            case _: print("an invalid ! input , pls enter a valid input...")
