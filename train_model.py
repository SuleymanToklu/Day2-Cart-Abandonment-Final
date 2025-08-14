import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains baseline and finetuned models and returns the champion model and results."""
    print("Training and evaluating models...")
    
    baseline_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    
    baseline_results = {
        'name': 'XGBoost (Baseline)',
        'recall': recall_score(y_test, y_pred_base),
        'precision': precision_score(y_test, y_pred_base),
        'f1': f1_score(y_test, y_pred_base),
        'cm': confusion_matrix(y_test, y_pred_base) 
    }
    print("\n--- Baseline Model Results ---")
    print(classification_report(y_test, y_pred_base))

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    tuned_model = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio
    )
    tuned_model.fit(X_train, y_train)
    y_pred_tuned = tuned_model.predict(X_test)

    tuned_results = {
        'name': 'XGBoost (Finetuned)',
        'recall': recall_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'f1': f1_score(y_test, y_pred_tuned),
        'cm': confusion_matrix(y_test, y_pred_tuned) 
    }
    print("\n--- Finetuned Model Results ---")
    print(classification_report(y_test, y_pred_tuned))
    
    return tuned_model, baseline_results, tuned_results

def save_artifacts(model, columns, results, project_name):
    """Saves the model, columns, and results to .pkl files."""
    print("\nSaving artifacts...")
    
    joblib.dump(model, f'{project_name}_model.pkl')
    joblib.dump(columns, f'{project_name}_columns.pkl')
    joblib.dump(results, f'{project_name}_results.pkl')

    print(f"✅ All artifacts for '{project_name}' saved successfully!")