import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

def create_confusion_matrix_plot(cm, title, filename):
    """Creates and saves a confusion matrix plot."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Terk Etti', 'Satın Aldı'], yticklabels=['Terk Etti', 'Satın Aldı'])
    plt.title(title)
    plt.ylabel('Gerçek Durum')
    plt.xlabel('Tahmin Edilen Durum')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {filename}")

def create_feature_importance_plot(model, columns, filename):
    """Creates and saves a feature importance plot."""
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, columns)), columns=['Value','Feature'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(10), palette='viridis')
    plt.title('Modelin En Önemli Gördüğü 10 Kriter')
    plt.xlabel('Önem Düzeyi')
    plt.ylabel('Özellikler')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved: {filename}")

def run_training_pipeline():
    print("1/4 - Loading data...")
    df = pd.read_csv("online_shoppers_intention.csv")
    
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
    baseline_cm = confusion_matrix(y_test, y_pred_base)
    create_confusion_matrix_plot(baseline_cm, 'Baseline Model Performansı', 'baseline_cm.png')
    
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    tuned_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
    tuned_model.fit(X_train, y_train)
    tuned_cm = confusion_matrix(y_test, tuned_model.predict(X_test))
    create_confusion_matrix_plot(tuned_cm, 'Finetuned Model Performansı', 'tuned_cm.png')
    create_feature_importance_plot(tuned_model, list(X.columns), 'feature_importance.png')

    print("4/4 - Saving artifacts...")
    joblib.dump(tuned_model, 'model.pkl')
    joblib.dump(list(X.columns), 'model_columns.pkl')
    print("✅ Artifacts saved successfully!")

if __name__ == "__main__":
    run_training_pipeline()