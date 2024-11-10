
# Titanic Survival Predictor

This project is a machine learning-based application that predicts a passenger's survival on the Titanic through a user-friendly graphical interface. It utilizes the Titanic dataset to train a Random Forest Classifier model, and Tkinter for the interactive GUI.

## Features

- **Data Handling**: Loads and preprocesses the Titanic dataset, addressing missing values and encoding categorical variables to prepare data for machine learning.
- **Survival Prediction**: Uses a trained Random Forest model to predict if a passenger would have survived, based on inputs such as passenger class, gender, age, fare, and embarkation port.
- **User Interface**: A graphical user interface, built with Tkinter, allows users to input data and receive predictions easily.

## Installation

### Prerequisites

- Python 3.6 or higher
- Required packages: pandas, scikit-learn, and Tkinter (Tkinter is included with Python on Windows)

### Setup

1. Clone the repository from GitHub.
2. Navigate to the project directory.
3. Install dependencies for pandas and scikit-learn.
4. Ensure your Titanic dataset is in the project directory, or update the dataset path in the code.

## Usage

To start the application, run the main Python file. Enter passenger details, including class, gender, age, and other relevant information, into the GUI fields. Click the **Predict Survival!** button to view the prediction.

### GUI Walkthrough

1. **Passenger Details**: Input the passenger’s class, gender, age, number of relatives on board, fare, and embarkation port.
2. **Prediction**: Click the predict button to see whether the model predicts survival for the input data.

### Application Workflow

- **Data Loading and Cleaning**: Reads and preprocesses the Titanic dataset, managing missing values and encoding categorical features.
- **Model Training**: Trains a Random Forest model using the cleaned dataset.
- **Prediction**: The GUI gathers passenger details, which are processed and sent to the model for survival prediction.
- **GUI**: Built using Tkinter, it allows users to input details and view results in a straightforward way.

## Troubleshooting

- Ensure the Titanic dataset includes the 'Survived' column for training.
- If dependencies are missing, use pip to install them as needed.

## License

This project is licensed under the MIT License.
