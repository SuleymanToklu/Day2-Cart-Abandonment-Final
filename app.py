import streamlit as st
import pandas as pd
import joblib

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Sepeti Terk Etme Tahmini", page_icon="🛒", layout="wide")

# --- Kaynakları Yükleme ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        return None, None

model, model_columns = load_resources()

# --- Ana Başlık ---
st.title("🛒 Alışveriş Sepeti Terk Etme Tahmini Projesi")
st.markdown("Bu interaktif uygulama, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını tahmin eder.")

# --- Sekmeli İçerik Yapısı ---
tab1, tab2 = st.tabs(["📊 **Proje Raporu**", "🧠 **Tahmin Aracı**"])

with tab1:
    st.header("Model Geliştirme Hikayesi: Finetuning'in Gücü")
    st.write("""
    İlk 'Baseline' model, satın alacak müşterileri yakalamada zayıf kaldı. Bu sorunu çözmek için, dengesiz veri setine özel bir hiperparametre (`scale_pos_weight`) ile model **Finetune** edildi ve müşteri yakalama oranı **%57'den %68'e** yükseltildi. Bu, daha fazla potansiyel satışın kurtarılması anlamına geliyor.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Önce: Baseline Model")
        st.image("baseline_cm.png", use_column_width=True)
    with col2:
        st.subheader("Sonra: Finetuned Model")
        st.image("tuned_cm.png", use_column_width=True)
    
    st.header("Modelin Karar Kriterleri")
    st.write("Model, bir müşterinin sepeti terk edip etmeyeceğine karar verirken bazı özelliklere diğerlerinden daha fazla önem verir. Aşağıda modelin en önemli bulduğu 10 kriteri görebilirsiniz:")
    st.image("feature_importance.png", use_column_width=True)

with tab2:
    st.header("Canlı Tahmin Aracı")
    st.write("Aşağıdaki slider ve seçenekleri değiştirerek farklı ziyaretçi profilleri için tahminler yapabilirsiniz.")
    
    if not model or not model_columns:
        st.error("Model dosyaları yüklenemedi. Lütfen önce `train_model.py`'yi çalıştırın.")
    else:
        # --- Kullanıcı Girdileri ---
        product_related = st.slider('Gezilen Ürün Sayfası', 0, 700, 30)
        exit_rates = st.slider('Çıkış Oranı', 0.0, 0.2, 0.04, format="%.4f")
        page_values = st.slider('Sayfa Değeri', 0.0, 362.0, 6.0, format="%.2f")
        month = st.selectbox('Ay', ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        visitor_type = st.selectbox('Ziyaretçi Tipi', ['Returning_Visitor', 'New_Visitor', 'Other'])

        if st.button('Tahmin Yap'):
            input_dict = {col: 0 for col in model_columns}
            input_dict['ProductRelated'] = product_related
            input_dict['ExitRates'] = exit_rates
            input_dict['PageValues'] = page_values
            
            month_col = 'Month_' + month
            if month_col in input_dict:
                input_dict[month_col] = 1
            
            visitor_col = 'VisitorType_' + visitor_type
            if visitor_col in input_dict:
                input_dict[visitor_col] = 1
                    
            input_df = pd.DataFrame([input_dict])[model_columns]
            
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.subheader('🔮 Tahmin Sonucu')
            if prediction[0] == 1:
                st.success(f"Bu ziyaretçinin SATIN ALMA olasılığı yüksek! (Olasılık: {prediction_proba[0][1]:.2%})")
            else:
                st.error(f"Bu ziyaretçinin SEPETİ TERK ETME olasılığı yüksek. (Satın Almama Olasılığı: {prediction_proba[0][0]:.2%})")