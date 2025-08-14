import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Proje Raporu", page_icon="🛒", layout="wide")

@st.cache_data
def load_results():
    """Karşılaştırma sonuçlarını yükler."""
    try:
        return joblib.load('day-2/cart_abandonment_results.pkl')
    except FileNotFoundError:
        return None

results = load_results()

st.title("🛒 Alışveriş Sepeti Terk Etme Tahmini Projesi")
st.markdown("Bu rapor, bir e-ticaret sitesi için geliştirilen müşteri terk etme tahmin modelinin detaylarını, geliştirme sürecini ve sonuçlarını içermektedir.")

tab1, tab2, tab3 = st.tabs(["🎯 **Proje Özeti**", "📊 **Model Performansı**", "🧠 **Teknik Detaylar**"])

with tab1:
    st.header("Projenin Amacı ve İş Değeri")
    st.image("https://i.imgur.com/v2d62pT.png", caption="Müşteri yolculuğu ve terk etme noktası")
    st.write("""
    Bu projenin temel amacı, bir online ziyaretçinin davranışlarını analiz ederek satın alma işlemini tamamlayıp tamamlamayacağını önceden tahmin etmektir.
    
    **Bu Neden Önemli?**
    - **Proaktif Müdahale:** Sepeti terk etme olasılığı yüksek olan müşterilere, siteyi terk etmeden önce özel indirimler veya yardım teklifleri sunulabilir.
    - **Pazarlama Optimizasyonu:** Hangi tür ziyaretçilerin daha çok satın alma yaptığını anlayarak, pazarlama kampanyaları daha verimli bir şekilde hedeflenebilir.
    """)
    st.info("Modeli canlı olarak test etmek için soldaki menüden **'🧠 Tahmin Araci'** sayfasına geçebilirsiniz.")

with tab2:
    st.header("Model Performansı: Baseline vs. Finetuned")
    
    if results:
        baseline = results['baseline']
        tuned = results['tuned']
        
        st.write("""
        İlk denenen 'Baseline' model, satın alacak müşterileri yakalamada zayıf kaldı. Bu sorunu çözmek için, dengesiz veri setine özel bir hiperparametre (`scale_pos_weight`) ile model **Finetune** edildi ve müşteri yakalama oranı (Recall) **%57'den %68'e** yükseltildi.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Önce: Baseline Model")
            st.image("baseline_cm.png", use_container_width=True)
            st.caption(f"""
            **Grafik Yorumu:** Bu model, satın alacak olan **{baseline['cm'][1][0] + baseline['cm'][1][1]}** müşteriden **{baseline['cm'][1][0]}** tanesini kaçırdı. 
            Bu, potansiyel satışların önemli bir kısmını gözden kaçırmak demektir.
            """)

        with col2:
            st.subheader("Sonra: Finetuned Model")
            st.image("tuned_cm.png", use_container_width=True)
            st.caption(f"""
            **Grafik Yorumu:** Finetuning sonrası, kaçırılan müşteri sayısı **{baseline['cm'][1][0]}'dan {tuned['cm'][1][0]}'a** düştü. 
            Model artık daha fazla potansiyel alıcıyı doğru tespit edebiliyor.
            """)

    else:
        st.error("Karşılaştırma sonuç dosyası bulunamadı. Lütfen önce `train_model.py`'yi çalıştırıp dosyaları GitHub'a gönderdiğinizden emin olun.")

with tab3:
    st.header("Teknik Detaylar ve Öğrenimler")
    
    with st.expander("🤔 Neden Sadece 'Accuracy' (Doğruluk) Yeterli Değil? - Tembel Doktor Analojisi"):
        st.write("""
        Bu projede olduğu gibi **dengesiz veri setlerinde** (yani bir sınıfın diğerinden çok daha fazla olduğu durumlarda), sadece doğruluk oranına bakmak yanıltıcı olabilir. 
        
        **Örnek:** 1000 hastadan sadece 1'inin hasta olduğu nadir bir hastalığı düşünelim. Hiçbir test yapmadan herkese "Sen hasta değilsin" diyen tembel bir doktor, **%99.9 doğruluk oranına** sahip olur. Ancak asıl görevi olan o 1 hastayı bulma işinde **%100 başarısızdır.**
        
        Bizim projemizde de müşterilerin sadece %15'i satın alma yapıyor. Bu yüzden, modelimizin ne kadar iyi olduğunu anlamak için **Precision** (keskinlik) ve özellikle **Recall** (müşteri yakalama oranı) gibi daha derin metriklere odaklandık. Bu yaklaşım, veri bilimi projelerinde standart bir en iyi pratiktir.
        """)

    st.subheader("Modelin Karar Kriterleri")
    st.write("Model, bir müşterinin sepeti terk edip etmeyeceğine karar verirken bazı özelliklere diğerlerinden daha fazla önem verir. Aşağıda modelin en önemli bulduğu 10 kriteri görebilirsiniz:")
    st.image("feature_importance.png", use_container_width=True)
    st.caption("Grafikten de anlaşıldığı üzere, bir kullanıcının 'Sayfa Değeri' (PageValues) ve siteden 'Çıkış Oranı' (ExitRates), satın alma niyetini belirlemede en güçlü sinyallerdir.")

