# wap to make a machine learning model

# all imported libraries and package
import os
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# define a method to return the name of the current directory
def current_directory():
    current_directory = os.getcwd()
    return current_directory

# define a method to select a csv file from the directory
def select_csv(directory_path):

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print("Error: Directory not found.")
        return None

    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter out only the CSV files
    csv_files = [file for file in files if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory.")
        return None

    # Display the list of CSV files to the user
    print("CSV files in the directory:")
    for i, csv_file in enumerate(csv_files, start=1):
        print(f"{i}. {csv_file}")

    # Ask the user to choose a file
    while True:
        choice = input("Enter the number of the CSV file you want to select (or 'q' to quit): ")
        if choice.lower() == 'q':
            return None
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(csv_files):
                selected_file = csv_files[choice_index]
                return selected_file
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


# define a method to load a data from csv into a dataframe
def read_csv_to_dataframe(csv_file_path):
    try:
        # Read the CSV file into a DataFrame
        dataframe = pd.read_csv(csv_file_path)
        return dataframe
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None
    except Exception as e:
        print("Error:", e)
        return None


# define a method to split a data into training and testing data
def split_data(dataframe, test_size=0.2, random_state=None):
    X = dataframe[['YearsExperience']]  # Features
    y = dataframe['Salary']        # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# define a method to create a machine learning model
def create_and_train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# define a method to predict a test data
def predict_test_data(model,test_data):
    y_pred = model.predict(test_data)
    return y_pred

# define a method to predict an actual data 
def predict_actual_data(X_train, y_train,data):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(data)
    return y_pred


# define a method to show predicted data
def show_predicted_data(predicted_data,y_test):
    print("Predicted\tActual")
    for i in range(len(predicted_data)):
        print(f"{predicted_data[i]}\t\t{y_test.iloc[i]}")


# define a method to check a model performance
def check_model_performance(y_test,predicted_test_data):
    r2=r2_score(y_test,predicted_test_data)
    print("Our model is %2.2f%% accurate" %(r2*100))

# main function
if __name__=="__main__":
    current_dir=current_directory()
    csv_file=select_csv(current_dir)
    dataframe = read_csv_to_dataframe(csv_file)
    if dataframe is not None:
        print("Data loaded successfully.")
        print(dataframe)
        print("\nData Created...")

    test_size=float(input("enter the test data size between ( 0 - 1 ) : "))
    X_train, X_test, y_train, y_test = split_data(dataframe, test_size=test_size, random_state=42)

    print("Model creation in progression")
    model=create_and_train_model(X_train, y_train)
    print("Model is created")
    
    input("Press ENTER key to predict test data in trained model")
    predicted_test_data=predict_test_data(model,X_test)
    show_predicted_data(predicted_test_data,y_test)
    check_model_performance(y_test,predicted_test_data)
        
