# pages/1_🧠_Tahmin_Araci.py

import streamlit as st
import pandas as pd
import joblib

# --- Kaynakları Yükleme ---
try:
    model = joblib.load('day-2/cart_abandonment_model.pkl')
    model_columns = joblib.load('day-2/cart_abandonment_columns.pkl')
except FileNotFoundError:
    st.error("Model dosyaları bulunamadı. Lütfen önce `train_model.py`'yi çalıştırıp dosyaları GitHub'a gönderdiğinizden emin olun.")
    st.stop()

st.title('🧠 Sepeti Terk Etme Tahmin Araci')

# --- Kullanıcı Girdileri ---
st.sidebar.header('Ziyaretçi Davranışları')
product_related = st.sidebar.slider('Gezilen Ürün Sayfası', 0, 700, 30)
exit_rates = st.sidebar.slider('Çıkış Oranı', 0.0, 0.2, 0.04, format="%.4f")
page_values = st.sidebar.slider('Sayfa Değeri', 0.0, 362.0, 6.0, format="%.2f")
month = st.sidebar.selectbox('Ay', ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
visitor_type = st.sidebar.selectbox('Ziyaretçi Tipi', ['Returning_Visitor', 'New_Visitor', 'Other'])

if st.sidebar.button('Tahmin Yap'):
    # --- Tahmin Mantığı ---
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

    # --- Sonuç Ekranı ---
    st.subheader('🔮 Tahmin Sonucu')
    if prediction[0] == 1:
        st.success(f"Bu ziyaretçinin SATIN ALMA olasılığı yüksek! (Olasılık: {prediction_proba[0][1]:.2%})")
    else:
        st.error(f"Bu ziyaretçinin SEPETİ TERK ETME olasılığı yüksek. (Satın Almama Olasılığı: {prediction_proba[0][0]:.2%})")