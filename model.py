import joblib
import os
import pandas as pd
from typing import Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def open_data(path: str = 'data/loan_approval.csv') -> pd.DataFrame:
    return pd.read_csv(path)


def save_model(model_name: str, model: Any, scaler: MinMaxScaler = None, label_encoders: dict = None) -> None:
    path = f'models/model_{model_name}.pkl'.replace(' ', '_')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model {model_name} saved to {path}")
    
    if scaler is not None:
        scaler_path = f'models/scaler_{model_name}.pkl'.replace(' ', '_')
        joblib.dump(scaler, scaler_path)
    
    if label_encoders is not None:
        encoders_path = f'models/encoders_{model_name}.pkl'.replace(' ', '_')
        joblib.dump(label_encoders, encoders_path)


def load_model(model_name: str) -> tuple[Any, MinMaxScaler, dict]:
    path = f'models/model_{model_name}.pkl'.replace(' ', '_')
    model = joblib.load(path)
    
    scaler_path = f'models/scaler_{model_name}.pkl'.replace(' ', '_')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    encoders_path = f'models/encoders_{model_name}.pkl'.replace(' ', '_')
    label_encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
    
    return model, scaler, label_encoders


# Function to detect outliers using IQR
def detect_outliers_iqr(data: pd.DataFrame, column: str):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def preprocess_data(data: pd.DataFrame, scaler: MinMaxScaler = None, label_encoders: dict = None, fit: bool = True) -> tuple[pd.DataFrame, MinMaxScaler, dict]:
    """
    Preprocess data. Returns (processed_data, scaler, label_encoders)
    If fit=True, creates new transformers. If fit=False, uses provided transformers.
    """
    data = data.copy()
    
    # Remove name and city columns if they exist
    if 'name' in data.columns:
        data.drop('name', axis=1, inplace=True)
    if 'city' in data.columns:
        data.drop('city', axis=1, inplace=True)
    
    has_target = 'loan_approved' in data.columns
    
    # Detect outliers only during training
    if fit:
        num_cols = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
        for col in num_cols:
            if col in data.columns:
                outliers, lb, ub = detect_outliers_iqr(data, col)
                print(f"📌 {col} → Outliers found: {len(outliers)} (Lower: {lb:.2f}, Upper: {ub:.2f})")

    # Encode loan_approved if exists (only during training)
    if has_target and fit:
        le_target = LabelEncoder()
        data['loan_approved'] = le_target.fit_transform(data['loan_approved'])

    # Encode categorical columns
    cat_cols = data.select_dtypes(include=['object', 'bool']).columns
    if has_target and 'loan_approved' in cat_cols:
        cat_cols = cat_cols.drop('loan_approved')
    
    if fit:
        print("📌 Categorical Columns:", list(cat_cols))
        label_encoders = {}
    
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
            print(f"✅ Encoded column: {col}")
        else:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen values
                data[col] = data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                data[col] = le.transform(data[col])

    # Scale numeric columns
    num_cols_to_scale = data.select_dtypes(include=['int64', 'float64']).columns
    if has_target and 'loan_approved' in num_cols_to_scale:
        num_cols_to_scale = num_cols_to_scale.drop('loan_approved')
    
    if len(num_cols_to_scale) > 0:
        if fit:
            scaler = MinMaxScaler()
            data[num_cols_to_scale] = scaler.fit_transform(data[num_cols_to_scale])
        else:
            if scaler is not None:
                data[num_cols_to_scale] = scaler.transform(data[num_cols_to_scale])
            else:
                raise ValueError("Scaler must be provided for prediction")
    
    return data, scaler, label_encoders


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop('loan_approved', axis=1)
    y = data['loan_approved']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def predict_model(model: Any, data: pd.DataFrame, scaler: MinMaxScaler = None, label_encoders: dict = None) -> Any:
    data_processed, _, _ = preprocess_data(data, scaler=scaler, label_encoders=label_encoders, fit=False)
    return model.predict(data_processed)


def main():
    data = open_data()
    data, scaler, label_encoders = preprocess_data(data, fit=True)
    X_train, X_test, y_train, y_test = split_data(data)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, acc))
        
        print(f"\n📌 Model: {name}")
        print("Accuracy:", acc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("-" * 100)
        save_model(name, model, scaler, label_encoders)

    best_model_name, best_accuracy = max(results, key=lambda x: x[1])
    print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

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
    model_name = "Logistic Regression"
    model, scaler, label_encoders = load_model(model_name)
    for customer in customers:
        predicted_approval = predict_model(model, customer, scaler, label_encoders)
        print(f"✅ Predicted Loan Approval: {'Approved' if predicted_approval[0] == 1 else 'Rejected'}")
        print(f"✅ Predicted Loan Approval Class: {predicted_approval[0]}")