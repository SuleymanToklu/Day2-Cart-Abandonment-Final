import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("Training started...")

df = pd.read_csv("online_shoppers_intention.csv")
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
df_processed = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

X = df_processed.drop('Revenue', axis=1)
y = df_processed['Revenue']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
model.fit(X, y)

joblib.dump(model, 'model.pkl')
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("✅ Model and columns saved successfully as 'model.pkl' and 'model_columns.pkl'")