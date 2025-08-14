import streamlit as st
import joblib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_model import run_training_pipeline

@st.cache_resource
def load_or_generate_resources():
    """
    Checks if model artifacts exist. If not, runs the training script to generate them.
    Then, loads and returns the artifacts.
    """
    results_path = 'day-2/cart_abandonment_results.pkl'
    
    if not os.path.exists(results_path):
        st.warning("Model dosyaları bulunamadı. Model şimdi eğitiliyor, bu işlem birkaç dakika sürebilir...")
        with st.spinner('Eğitim süreci çalıştırılıyor...'):
            success = run_training_pipeline()
            if not success:
                st.error("Model eğitimi sırasında bir hata oluştu. Lütfen logları kontrol edin.")
                return None, None
    
    try:
        results = joblib.load(results_path)
        return results
    except FileNotFoundError:
        return None

st.set_page_config(page_title="Proje Raporu", page_icon="🛒", layout="wide")
st.title("🛒 Alışveriş Sepeti Terk Etme Tahmini Projesi")

results = load_or_generate_resources()

if results:
    tab1, tab2 = st.tabs(["🎯 **Proje Özeti**", "📊 **Model Performansı**"])

    with tab1:
        st.header("Projenin Amacı ve İş Değeri")
        st.write("Bu proje, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını önceden tahmin etmeyi amaçlar.")
        st.info("Modeli canlı olarak test etmek için soldaki menüden **'🧠 Tahmin Araci'** sayfasına geçebilirsiniz.")

    with tab2:
        st.header("Model Performansı: Baseline vs. Finetuned")
        baseline = results['baseline']
        tuned = results['tuned']
        
        st.write("İlk 'Baseline' model, satın alacak müşterileri yakalamada zayıf kaldı (**Recall: %57**). Bu sorunu çözmek için, `scale_pos_weight` hiperparametresi ile model **Finetune** edildi ve müşteri yakalama oranı **%68'e** yükseltildi.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Önce: Baseline Model")
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(baseline['cm'], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            st.pyplot(fig)

        with col2:
            st.subheader("Sonra: Finetuned Model")
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(tuned['cm'], annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
            st.pyplot(fig)
else:
    st.error("Kaynaklar yüklenemedi. Lütfen uygulamanın loglarını kontrol edin.")