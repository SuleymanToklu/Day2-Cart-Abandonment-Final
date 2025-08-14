import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

def run_training_pipeline():
    """
    Loads data, preprocesses, trains models, and saves all necessary artifacts.
    This is the main function to be called by other scripts.
    """
    print("1/4 - Loading data...")
    try:
        df = pd.read_csv("day-2/online_shoppers_intention.csv")
    except FileNotFoundError:
        print("HATA: 'day-2/online_shoppers_intention.csv' bulunamadı.")
        return False

    print("2/4 - Preprocessing data...")
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    df_processed = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)
    
    X = df_processed.drop('Revenue', axis=1)
    y = df_processed['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("3/4 - Training and evaluating models...")
    baseline_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    baseline_results = {
        'recall': recall_score(y_test, y_pred_base), 'precision': precision_score(y_test, y_pred_base),
        'f1': f1_score(y_test, y_pred_base), 'cm': confusion_matrix(y_test, y_pred_base)
    }
    
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    tuned_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
    tuned_model.fit(X_train, y_train)
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_results = {
        'recall': recall_score(y_test, y_pred_tuned), 'precision': precision_score(y_test, y_pred_tuned),
        'f1': f1_score(y_test, y_pred_tuned), 'cm': confusion_matrix(y_test, y_pred_tuned)
    }

    print("4/4 - Saving artifacts...")
    project_name = "cart_abandonment"
    save_dir = "day-2"
    
    joblib.dump(tuned_model, os.path.join(save_dir, f'{project_name}_model.pkl'))
    model_columns = list(X.columns)
    joblib.dump(model_columns, os.path.join(save_dir, f'{project_name}_columns.pkl'))
    comparison_results = {'baseline': baseline_results, 'tuned': tuned_results}
    joblib.dump(comparison_results, os.path.join(save_dir, f'{project_name}_results.pkl'))
    
    print("✅ Training pipeline completed successfully!")
    return True

if __name__ == "__main__":
    run_training_pipeline()