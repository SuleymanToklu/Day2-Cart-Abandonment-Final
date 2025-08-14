# 0_🛒_Proje_Raporu.py

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Proje Raporu", page_icon="🛒", layout="wide")

# --- Kaynakları Yükleme ---
try:
    results = joblib.load('day-2/cart_abandonment_results.pkl')
except FileNotFoundError:
    results = None

st.title("🛒 Alışveriş Sepeti Terk Etme Tahmini Projesi")
st.markdown("---")

# --- Sekmeli İçerik Yapısı ---
tab1, tab2 = st.tabs(["🎯 **Proje Özeti**", "📊 **Model Performansı**"])

with tab1:
    st.header("Projenin Amacı ve İş Değeri")
    st.write("""
    Bu projenin temel amacı, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını önceden tahmin etmektir.
    Modeli canlı olarak test etmek için soldaki menüden **'🧠 Tahmin Araci'** sayfasına geçebilirsiniz.
    """)

with tab2:
    st.header("Model Performansı: Baseline vs. Finetuned")
    
    if results:
        baseline = results['baseline']
        tuned = results['tuned']
        
        st.write("""
        İlk 'Baseline' model, satın alacak müşterileri yakalamada zayıf kaldı (**Recall: %57**). 
        Bu sorunu çözmek için, `scale_pos_weight` hiperparametresi ile model **Finetune** edildi ve müşteri yakalama oranı **%68'e** yükseltildi.
        """)
        
        # Metrikleri bir DataFrame'e dönüştürme
        metrics_df = pd.DataFrame({
            'Baseline Model': [f"{baseline['precision']:.2%}", f"{baseline['recall']:.2%}", f"{baseline['f1']:.2f}"],
            'Finetuned Model': [f"{tuned['precision']:.2%}", f"{tuned['recall']:.2%}", f"{tuned['f1']:.2f}"]
        }, index=['Precision (Kesinlik)', 'Recall (Yakalama)', 'F1-Score'])
        
        st.subheader("Metriklerin Karşılaştırması")
        st.dataframe(metrics_df)

    else:
        st.error("Karşılaştırma sonuç dosyası bulunamadı. Lütfen önce `train_model.py`'yi çalıştırıp dosyaları GitHub'a gönderdiğinizden emin olun.")