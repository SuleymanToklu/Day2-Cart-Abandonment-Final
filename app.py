import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
import warnings

st.set_page_config(page_title="Sepeti Terk Etme Tahmini", page_icon="🛒", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)

@st.cache_resource
def train_and_prepare_resources():
    """
    Veriyi yükler, işler, baseline ve finetuned modelleri eğitir.
    Gerekli tüm sonuçları ve en iyi modeli hafızada tutmak için döndürür.
    Bu fonksiyon SADECE BİR KEZ çalışır.
    """
    df = pd.read_csv("online_shoppers_intention.csv")
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    df_processed = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)
    
    X = df_processed.drop('Revenue', axis=1)
    y = df_processed['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    baseline_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    baseline_results = {
        'Precision': precision_score(y_test, y_pred_base),
        'Recall': recall_score(y_test, y_pred_base),
        'F1-Score': f1_score(y_test, y_pred_base)
    }
    
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    tuned_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
    tuned_model.fit(X_train, y_train)
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_results = {
        'Precision': precision_score(y_test, y_pred_tuned),
        'Recall': recall_score(y_test, y_pred_tuned),
        'F1-Score': f1_score(y_test, y_pred_tuned)
    }
    
    feature_imp = pd.DataFrame(data=tuned_model.feature_importances_, index=list(X.columns), columns=['Value'])
    feature_imp = feature_imp.sort_values(by="Value", ascending=False).head(10)

    return tuned_model, list(X.columns), baseline_results, tuned_results, feature_imp

st.title("🛒 Alışveriş Sepeti Terk Etme Tahmini Projesi")

with st.spinner('Model ve kaynaklar yükleniyor... Bu işlem ilk çalıştırmada biraz sürebilir.'):
    model, model_columns, baseline_results, tuned_results, feature_imp = train_and_prepare_resources()

st.success("Model başarıyla yüklendi ve kullanıma hazır!")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 **Proje Raporu**", "🧠 **Tahmin Aracı**", "🔧 **Teknik Detaylar**"])

with tab1:
    st.header("Projenin Amacı ve İş Değeri")
    st.write("Bu proje, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını önceden tahmin etmeyi amaçlar.")
    
    st.header("Model Performansı: Baseline vs. Finetuned")
    st.write("İlk 'Baseline' model, satın alacak müşterileri yakalamada zayıf kaldı (**Recall: 57%**). Bu sorunu çözmek için, `scale_pos_weight` hiperparametresi ile model **Finetune** edildi ve müşteri yakalama oranı **68%'e** yükseltildi.")
    
    metrics_df = pd.DataFrame({'Baseline Model': baseline_results, 'Finetuned Model': tuned_results})
    st.dataframe(metrics_df.style.format("{:.2%}"))

with tab2:
    st.header("Canlı Tahmin Aracı")
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

with tab3:
    st.header("Teknik Detaylar ve Öğrenimler")
    with st.expander("🤔 Neden Sadece 'Accuracy' Yeterli Değil? - Tembel Doktor Analojisi"):
        st.write("""
        Bu projede olduğu gibi **dengesiz veri setlerinde**, sadece doğruluk oranına bakmak yanıltıcı olabilir. 
        **Örnek:** 1000 hastadan sadece 1'inin hasta olduğu nadir bir hastalığı düşünelim. Hiçbir test yapmadan herkese "Sen hasta değilsin" diyen tembel bir doktor, **%99.9 doğruluk oranına** sahip olur. Ancak asıl görevi olan o 1 hastayı bulma işinde **%100 başarısızdır.**
        Bizim projemizde de müşterilerin sadece %15'i satın alma yapıyor. Bu yüzden, **Recall** (müşteri yakalama oranı) gibi daha derin metriklere odaklandık.
        """)

    st.subheader("Modelin Karar Kriterleri")
    st.write("Aşağıda modelin, tahmin yaparken en çok önem verdiği 10 kriteri görebilirsiniz:")
    st.bar_chart(feature_imp)