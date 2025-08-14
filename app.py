import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sepeti Terk Etme Tahmini", page_icon="🛒", layout="wide")

@st.cache_resource
def load_resources():
    """Model ve ilgili dosyaları yükler."""
    try:
        model = joblib.load('cart_abandonment_model.pkl')
        model_columns = joblib.load('cart_abandonment_columns.pkl')
        results = joblib.load('cart_abandonment_results.pkl')
        return model, model_columns, results
    except FileNotFoundError:
        return None, None, None

model, model_columns, results = load_resources()

st.title("🛒 Alışveriş Sepeti Terk Etme Tahmini Projesi")
st.markdown("Bu interaktif uygulama, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını tahmin eder.")

# --- Hata Kontrolü 
if not all([model, model_columns, results]):
    st.error("Model dosyaları bulunamadı. Lütfen önce `train_model.py`'yi çalıştırıp dosyaları GitHub'a gönderdiğinizden emin olun.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎯 **Proje Raporu**", "🧠 **Tahmin Aracı**", "🔧 **Teknik Detaylar**"])

with tab1:
    st.header("Projenin Amacı ve İş Değeri")
    st.image("https://i.imgur.com/6Q5Z2Xk.png", caption="Müşteri yolculuğu ve terk etme noktası")
    st.write("""
    Bu projenin temel amacı, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını önceden tahmin etmektir.
    Bu, e-ticaret şirketlerinin potansiyel satış kayıplarını önceden tespit ederek özel indirimler veya hatırlatmalarla bu müşterileri geri kazanmalarına yardımcı olabilir.
    """)

with tab2:
    st.header("Model Performansı: Baseline vs. Finetuned")
    st.write("""
    İlk 'Baseline' model, satın alacak müşterileri yakalamada zayıf kaldı. Bu sorunu çözmek için, dengesiz veri setine özel bir hiperparametre (`scale_pos_weight`) ile model **Finetune** edildi ve müşteri yakalama oranı (Recall) **%57'den %68'e** yükseltildi.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Önce: Baseline Model")
        st.image("baseline_cm.png", use_container_width=True)
        st.caption(f"Bu model, satın alacak olan **{results['baseline']['cm'][1][0] + results['baseline']['cm'][1][1]}** müşteriden **{results['baseline']['cm'][1][0]}** tanesini kaçırdı.")

    with col2:
        st.subheader("Sonra: Finetuned Model")
        st.image("tuned_cm.png", use_container_width=True)
        st.caption(f"Finetuning sonrası, kaçırılan müşteri sayısı **{results['tuned']['cm'][1][0]}'a** düştü. Model artık daha fazla potansiyel alıcıyı doğru tespit edebiliyor.")

with tab3:
    st.header("Modelin Karar Kriterleri")
    st.write("Model, bir müşterinin sepeti terk edip etmeyeceğine karar verirken bazı özelliklere diğerlerinden daha fazla önem verir. Aşağıda modelin en önemli bulduğu 10 kriteri görebilirsiniz:")
    st.image("feature_importance.png", use_container_width=True)
    st.caption("Grafikten de anlaşıldığı üzere, bir kullanıcının 'Sayfa Değeri' (PageValues) ve siteden 'Çıkış Oranı' (ExitRates), satın alma niyetini belirlemede en güçlü sinyallerdir.")

st.markdown("---")
st.header("🧠 Canlı Tahmin Aracı")
st.write("Aşağıdaki slider ve seçenekleri değiştirerek farklı ziyaretçi profilleri için tahminler yapabilirsiniz.")

with st.form(key='prediction_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        product_related = st.slider('Gezilen Ürün Sayfası', 0, 700, 30)
    with col2:
        exit_rates = st.slider('Çıkış Oranı', 0.0, 0.2, 0.04, format="%.4f")
    with col3:
        page_values = st.slider('Sayfa Değeri', 0.0, 362.0, 6.0, format="%.2f")
    
    col4, col5 = st.columns(2)
    with col4:
        month = st.selectbox('Ay', ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    with col5:
        visitor_type = st.selectbox('Ziyaretçi Tipi', ['Returning_Visitor', 'New_Visitor', 'Other'])
    
    submit_button = st.form_submit_button(label='Tahmin Yap')

if submit_button:
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
