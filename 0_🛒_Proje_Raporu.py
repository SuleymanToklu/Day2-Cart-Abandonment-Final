import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="Proje Raporu", page_icon="🛒", layout="wide")

# --- Model Training Function ---
@st.cache_resource
def train_model():
    """
    Loads data, preprocesses, trains the model, and returns all necessary artifacts.
    This function runs ONLY ONCE and its result is cached.
    """
    # 1. Load Data
    df = pd.read_csv('day-2/online_shoppers_intention.csv')
    
    # 2. Preprocess Data
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    df_processed = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)
    
    X = df_processed.drop('Revenue', axis=1)
    y = df_processed['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Train the Final Model
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
    model.fit(X_train, y_train)
    
    # 4. Return artifacts
    return model, list(X.columns)

# --- Main App ---
st.title("🛒 Alışveriş Sepeti Terk Etme Tahmini Projesi")

with st.spinner('Model ve kaynaklar yükleniyor... Bu işlem ilk çalıştırmada biraz sürebilir.'):
    model, model_columns = train_model()

if model and model_columns:
    st.success("Model başarıyla yüklendi ve kullanıma hazır!")
    # Store the loaded resources in session state to share with other pages
    st.session_state['model'] = model
    st.session_state['model_columns'] = model_columns
else:
    st.error("Model yüklenirken bir hata oluştu. Lütfen logları kontrol edin.")
    st.stop()

st.markdown("---")
st.header("📖 Projenin Amacı")
st.write("Bu proje, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını tahmin etmeyi amaçlar.")
st.info("Modeli canlı olarak test etmek için soldaki menüden **'🧠 Tahmin Araci'** sayfasına geçebilirsiniz.")