import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tahmin Aracı", page_icon="🧠", layout="wide")

@st.cache_resource
def load_resources():
    """Model ve ilgili dosyaları yükler."""

    project_dir = os.path.join(os.path.dirname(__file__), '..')
    
    try:
        model = joblib.load(os.path.join(project_dir, 'cart_abandonment_model.pkl'))
        model_columns = joblib.load(os.path.join(project_dir, 'cart_abandonment_columns.pkl'))
        return model, model_columns
    except FileNotFoundError:
        return None, None

model, model_columns = load_resources()

st.title('🧠 Sepeti Terk Etme Tahmin Aracı')
st.info("Bu araç, bir ziyaretçinin davranışlarına göre satın alma yapıp yapmayacağını tahmin etmek için eğitilmiş bir XGBoost modeli kullanır.")

if not model or not model_columns:
    st.error("Model dosyaları yüklenemedi. Lütfen önce `train_model.py` script'ini çalıştırdığınızdan emin olun.")
    st.stop()

st.sidebar.header('Ziyaretçi Davranışları')
with st.sidebar.form(key='prediction_form'):
    product_related = st.slider('Gezilen Ürün Sayfası Sayısı', 0, 700, 30)
    exit_rates = st.slider('Çıkış Oranı (Exit Rate)', 0.0, 0.2, 0.04, format="%.4f")
    page_values = st.slider('Sayfa Değeri (Page Value)', 0.0, 362.0, 6.0, format="%.2f")
    month = st.selectbox('Ay', ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    visitor_type = st.selectbox('Ziyaretçi Tipi', ['Returning_Visitor', 'New_Visitor', 'Other'])
    
    submit_button = st.form_submit_button(label='Tahmin Yap')

if submit_button:
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0  

    input_df['ProductRelated'] = product_related
    input_df['ExitRates'] = exit_rates
    input_df['PageValues'] = page_values
    
    month_col = 'Month_' + month
    if month_col in input_df.columns:
        input_df[month_col] = 1
    
    visitor_col = 'VisitorType_' + visitor_type
    if visitor_col in input_df.columns:
        input_df[visitor_col] = 1
    
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('🔮 Tahmin Sonucu')
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.success("Bu ziyaretçinin **SATIN ALMA** olasılığı yüksek!")
        else:
            st.error("Bu ziyaretçinin **SEPETİ TERK ETME** olasılığı yüksek.")

    with col2:
        prob_to_purchase = prediction_proba[0][1]
        st.metric(label="Satın Alma Olasılığı", value=f"{prob_to_purchase:.2%}")
    
    st.progress(prob_to_purchase, text=f"Olasılık Skoru: {prob_to_purchase:.0%}")
    
    with st.expander("📊 Modelin En Önemli Gördüğü Kriterler"):
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, model_columns)), columns=['Value','Feature'])
        st.write("Model, tahmin yaparken en çok bu özelliklere dikkat ediyor:")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(10), ax=ax, palette='viridis')
        plt.title('En Önemli 10 Özellik')
        plt.xlabel('Önem Düzeyi')
        plt.ylabel('Özellikler')
        st.pyplot(fig)