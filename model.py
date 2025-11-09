import joblib
import os
import pandas as pd
from typing import Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def open_data(path: str = 'data/loan_approval.csv') -> pd.DataFrame:
    return pd.read_csv(path)

def save_model(model: Any, path: str = 'models/loan_approval.pkl') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path: str = 'models/loan_approval.pkl') -> Any:
    return joblib.load(path)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if 'name' in data.columns:
        data.drop('name', axis=1, inplace=True)
    if 'city' in data.columns:
        data.drop('city', axis=1, inplace=True)
    if 'loan_approved' in data.columns:
        le = LabelEncoder()
        data['loan_approved'] = le.fit_transform(data['loan_approved'])
    return data

def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop('loan_approved', axis=1)
    y = data['loan_approved']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    return accuracy_score(y_test, model.predict(X_test))

def predict_model(model: Any, data: pd.DataFrame) -> Any:
    return model.predict(data)

def main():
    data = open_data()
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    predicted_approval = predict_model(model, X_test)
    print(f"✅ Predicted Approval: {'Approved' if predicted_approval[0] == 1 else 'Rejected'}")
    print(f"✅ Predicted Approval Class: {predicted_approval[0]}")
    save_model(model, 'models/loan_approval.pkl')

if __name__ == '__main__':
    main()
    print('-' * 80)

    # Predicting the loan approval for a new customer
    customers = [
        pd.DataFrame({
            'income': [50000],
            'credit_score': [70],
            'loan_amount': [100000],
            'years_employed': [5],
            'points': [500]
        }),
        pd.DataFrame({
            'income': [50000],
            'credit_score': [700],
            'loan_amount': [100000],
            'years_employed': [5],
            'points': [50]
        })
    ]
    model = load_model()
    for customer in customers:
        new_customer = preprocess_data(customer)
        predicted_approval = predict_model(model, new_customer)
        print(f"✅ Predicted Loan Approval: {'Approved' if predicted_approval[0] == 1 else 'Rejected'}")
        print(f"✅ Predicted Loan Approval Class: {predicted_approval[0]}")