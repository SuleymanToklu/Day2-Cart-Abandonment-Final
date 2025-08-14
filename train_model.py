import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_and_save_model():
    print("1/3 - Loading data...")
    df = pd.read_csv("online_shoppers_intention.csv")

    print("2/3 - Preprocessing data and training model...")
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    df_processed = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

    X = df_processed.drop('Revenue', axis=1)
    y = df_processed['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
    model.fit(X_train, y_train)

    print("\n--- Model Performance ---")
    print(classification_report(y_test, model.predict(X_test)))

    print("3/3 - Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    print("✅ Artifacts saved successfully!")

if __name__ == "__main__":
    train_and_save_model()