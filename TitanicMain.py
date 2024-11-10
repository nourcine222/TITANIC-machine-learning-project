import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data, is_new_data=False):
    # Fill missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Fill 'Embarked' only if it's the training data (not new data)
    if not is_new_data:
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Drop unnecessary columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data = data.drop([col for col in columns_to_drop if col in data.columns], axis=1)

    # Encode 'Sex'
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])

    # One-hot encoding for 'Embarked'
    if 'Embarked' in data.columns:
        data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
    else:
        # If Embarked is missing, we add placeholder columns (only applies for prediction)
        if is_new_data:
            data['Embarked_Q'] = 0
            data['Embarked_S'] = 0

    if is_new_data:
        # Ensure required columns for predictions
        required_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_Q', 'Embarked_S']
        for col in required_columns:
            if col not in data.columns:
                data[col] = 0  # Fill missing columns with 0
        data = data[required_columns]
    else:
        # Ensure the 'Survived' column is present for training
        if 'Survived' not in data.columns:
            raise ValueError("The 'Survived' column is missing from the training data.")
    
    return data
def predict_survival(model, new_data):
    new_data_cleaned = clean_data(new_data, is_new_data=True)

    # Get the list of columns used during model fitting
    # Use model.feature_names_in_ for sklearn >= 1.0
    if hasattr(model, 'feature_names_in_'):
        training_columns = model.feature_names_in_
    else:
        # If model does not have 'feature_names_in_' (older sklearn versions)
        training_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_Q', 'Embarked_S']

    # Reindex new data columns to match the training data
    new_data_cleaned = new_data_cleaned.reindex(columns=training_columns, fill_value=0)

    # Ensure that columns match exactly (for debugging)
    if list(new_data_cleaned.columns) != list(training_columns):
        raise ValueError(f"Column mismatch! Expected {training_columns}, but got {list(new_data_cleaned.columns)}")

    # Make predictions
    predictions = model.predict(new_data_cleaned)
    return predictions


def create_ui(model):
    def submit():
        try:
            # Get input values
            pclass = int(pclass_entry.get())
            sex = sex_combobox.get()
            age = float(age_entry.get())
            sibsp = int(sibsp_entry.get())
            parch = int(parch_entry.get())
            fare = float(fare_entry.get())
            embarked = embarked_combobox.get()

            # Map inputs to the format needed for prediction
            sex_numeric = 1 if sex == 'Male' else 0

            # One-hot encode the 'Embarked' feature manually
            if embarked == 'C':
                embarked_Q = 0
                embarked_S = 0
            elif embarked == 'Q':
                embarked_Q = 1
                embarked_S = 0
            else:  # embarked == 'S'
                embarked_Q = 0
                embarked_S = 1

            # Create a DataFrame for the new passenger data with one-hot encoded 'Embarked' columns
            new_data = pd.DataFrame({
               'Pclass': [pclass],
            'Sex': [sex_numeric],
               'Age': [age],
            'SibSp': [sibsp],
                'Parch': [parch],
            'Fare': [fare],
                'Embarked_Q': [embarked_Q],
                'Embarked_S': [embarked_S]  # Embarked_C is implicit as 0 due to drop_first=True in get_dummies
            })

            # Get the prediction
            predictions = predict_survival(model, new_data)
            messagebox.showinfo("Survival Prediction", f"The prediction is: {'Survived' if predictions[0] == 1 else 'Did not survive'}")
        except Exception as e:
            print("Error", str(e))


    # Create main window
    window = tk.Tk()
    window.title("Titanic Survival Predictor")
    
    window.configure(bg="#f0f0f0")

    # Title label
    title_label = tk.Label(window, text="🌊 Titanic Survival Predictor 🌊", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
    title_label.pack(pady=10)

    # Input fields with fun labels
    tk.Label(window, text="🎫 Class (1, 2, or 3):", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=5)
    pclass_entry = tk.Entry(window, font=("Helvetica", 14))
    pclass_entry.pack(pady=5)

    tk.Label(window, text="👤 Gender (Male/Female):", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=5)
    sex_combobox = ttk.Combobox(window, values=["Male", "Female"], font=("Helvetica", 14))
    sex_combobox.pack(pady=5)

    tk.Label(window, text="🧑‍🤝‍🧑 Age:", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=5)
    age_entry = tk.Entry(window, font=("Helvetica", 14))
    age_entry.pack(pady=5)

    tk.Label(window, text="👪 Siblings/Spouses aboard:", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=5)
    sibsp_entry = tk.Entry(window, font=("Helvetica", 14))
    sibsp_entry.pack(pady=5)

    tk.Label(window, text="👶 Parents/Children aboard:", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=5)
    parch_entry = tk.Entry(window, font=("Helvetica", 14))
    parch_entry.pack(pady=5)

    tk.Label(window, text="💰 Fare:", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=5)
    fare_entry = tk.Entry(window, font=("Helvetica", 14))
    fare_entry.pack(pady=5)

    tk.Label(window, text="🚢 Embarked (C, Q, S):", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=5)
    embarked_combobox = ttk.Combobox(window, values=["C", "Q", "S"], font=("Helvetica", 14))
    embarked_combobox.pack(pady=5)

    # Submit button
    submit_button = tk.Button(window, text="🚀 Predict Survival!", command=submit, font=("Helvetica", 14), bg="#007BFF", fg="white")
    submit_button.pack(pady=20)

    window.mainloop()

# Load dataset and train model
if __name__ == "__main__":
    try:
        data = load_data('Titanic-Dataset.csv')  # Replace with your dataset path
        cleaned_data = clean_data(data)
        
        if 'Survived' not in cleaned_data.columns:
            raise ValueError("The 'Survived' column is missing from the cleaned data.")

        X = cleaned_data.drop('Survived', axis=1)
        y = cleaned_data['Survived']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Create the UI
        create_ui(model)
    except Exception as e:
        print(f"An error occurred: {e}")
