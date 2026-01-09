import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# =============================================
# LOAD MODEL DAN SCALER
# =============================================
@st.cache_resource
def load_model():
    model = joblib.load('models/fraud_detection_model.joblib')
    amount_scaler = joblib.load('models/amount_scaler.joblib')
    time_scaler = joblib.load('models/time_scaler.joblib')
    config = joblib.load('models/model_config.joblib')
    return model, amount_scaler, time_scaler, config

model, amount_scaler, time_scaler, config = load_model()

# =============================================
# FUNGSI PREDIKSI
# =============================================
def predict_fraud(transaction_data, threshold=None):
    if threshold is None:
        threshold = config['optimal_threshold']
    
    df_input = pd.DataFrame([transaction_data])
    
    df_input['Amount_scaled'] = amount_scaler.transform(df_input[['Amount']])
    df_input['Time_scaled'] = time_scaler.transform(df_input[['Time']])
    df_input = df_input.drop(['Amount', 'Time'], axis=1)
    df_input = df_input[config['feature_columns']]
    
    prob = model.predict_proba(df_input)[0, 1]
    prediction = 1 if prob >= threshold else 0
    
    return prediction, prob

# =============================================
# LOAD SAMPLE DATA
# =============================================
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('data/creditcard.csv')
        return df
    except FileNotFoundError:
        return None

df = load_sample_data()

# Check if dataset is available
dataset_available = df is not None

# =============================================
# SIDEBAR
# =============================================
st.sidebar.title("Pengaturan")
threshold = st.sidebar.slider(
    "Threshold Prediksi",
    min_value=0.1,
    max_value=0.9,
    value=config['optimal_threshold'],
    step=0.05,
    help="Threshold lebih rendah = lebih sensitif mendeteksi fraud"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tentang Model")
st.sidebar.markdown(f"""
- **Algorithm**: Random Forest
- **Precision**: 94.12%
- **Recall**: 81.63%
- **F1-Score**: 87.43%
- **Optimal Threshold**: {config['optimal_threshold']}
""")

# =============================================
# HEADER
# =============================================
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Sistem deteksi transaksi kartu kredit yang mencurigakan menggunakan Machine Learning.")

if not dataset_available:
    st.info("ℹ️ **Mode Demo Terbatas:** Dataset tidak tersedia. Gunakan tab 'Upload CSV' untuk prediksi.")

st.markdown("---")

# =============================================
# TAB NAVIGATION
# =============================================
tab1, tab2, tab3, tab4 = st.tabs(["Demo Prediksi", "Upload CSV", "Cara Penggunaan", "Tentang Project"])

# =============================================
# TAB 1: DEMO PREDIKSI
# =============================================
with tab1:
    st.header("Demo Prediksi")
    
    if not dataset_available:
        st.warning("⚠️ Dataset tidak tersedia. Fitur demo sample memerlukan file `data/creditcard.csv`")
        st.info("""
        **Untuk menggunakan fitur demo:**
        1. Download dataset dari [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
        2. Letakkan file `creditcard.csv` di folder `data/`
        3. Restart aplikasi
        
        **Alternatif:** Gunakan tab "Upload CSV" untuk prediksi dengan data Anda sendiri.
        """)
    else:
        st.markdown("Pilih sample transaksi untuk melihat hasil prediksi model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_type = st.selectbox(
                "Pilih Jenis Sample",
                ["Random Normal", "Random Fraud", "Random Transaksi"]
            )
            
            if st.button("Generate Sample", use_container_width=True):
                if sample_type == "Random Normal":
                    sample = df[df['Class'] == 0].sample(1).iloc[0]
                elif sample_type == "Random Fraud":
                    sample = df[df['Class'] == 1].sample(1).iloc[0]
                else:
                    sample = df.sample(1).iloc[0]
                
                st.session_state['current_sample'] = sample
        
        with col2:
            if 'current_sample' in st.session_state:
                sample = st.session_state['current_sample']
                actual_label = "FRAUD" if sample['Class'] == 1 else "NORMAL"
                
                st.markdown(f"**Amount:** ${sample['Amount']:.2f}")
                st.markdown(f"**Time:** {sample['Time']:.0f} detik")
                st.markdown(f"**Label Asli:** {actual_label}")
        
        st.markdown("---")
        
        if 'current_sample' in st.session_state:
            sample = st.session_state['current_sample']
            
            # Siapkan data untuk prediksi
            transaction = sample.drop('Class').to_dict()
            
            # Prediksi
            prediction, probability = predict_fraud(transaction, threshold)
            
            # Tampilkan hasil
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("### FRAUD DETECTED")
                else:
                    st.success("### NORMAL")
            
            with col2:
                st.metric("Probabilitas Fraud", f"{probability*100:.2f}%")
            
            with col3:
                st.metric("Threshold", f"{threshold*100:.0f}%")
            
            # Progress bar probabilitas
            st.markdown("**Confidence Level:**")
            st.progress(probability)
            
            # Detail V features (collapsible)
            with st.expander("Lihat Detail Fitur"):
                v_features = {k: v for k, v in transaction.items() if k.startswith('V')}
                v_df = pd.DataFrame([v_features])
                st.dataframe(v_df, use_container_width=True)

# =============================================
# TAB 2: UPLOAD CSV
# =============================================
with tab2:
    st.header("Batch Prediction")
    st.markdown("Upload file CSV untuk melakukan prediksi pada banyak transaksi sekaligus.")
    
    # Warning untuk format dataset
    st.warning("""
    ⚠️ **PENTING - Format Dataset Khusus:**
    
    Model ini **HANYA** dapat memprediksi dataset dari [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).
    
    CSV Anda **HARUS** memiliki kolom:
    - `Time` (waktu transaksi dalam detik)
    - `V1` sampai `V28` (fitur hasil PCA)
    - `Amount` (jumlah transaksi)
    - `Class` (opsional - akan diabaikan)
    
    ❌ **TIDAK BISA** digunakan untuk dataset transaksi kartu kredit biasa dengan format lain.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"**Total transaksi:** {len(df_upload):,}")
            
            # Cek kolom yang dibutuhkan
            required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"Kolom tidak lengkap. Kurang: {missing_cols}")
            else:
                if st.button("Jalankan Prediksi", use_container_width=True):
                    with st.spinner("Memproses..."):
                        results = []
                        for idx, row in df_upload.iterrows():
                            transaction = row.to_dict()
                            if 'Class' in transaction:
                                del transaction['Class']
                            pred, prob = predict_fraud(transaction, threshold)
                            results.append({
                                'Index': idx,
                                'Amount': row['Amount'],
                                'Prediction': 'FRAUD' if pred == 1 else 'NORMAL',
                                'Fraud_Probability': prob
                            })
                        
                        results_df = pd.DataFrame(results)
                        
                        # Summary
                        fraud_count = sum(results_df['Prediction'] == 'FRAUD')
                        normal_count = sum(results_df['Prediction'] == 'NORMAL')
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Transaksi", len(results_df))
                        col2.metric("Terdeteksi Fraud", fraud_count)
                        col3.metric("Normal", normal_count)
                        
                        # Tabel hasil
                        st.markdown("### Hasil Prediksi")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download hasil
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Hasil (CSV)",
                            csv,
                            "fraud_predictions.csv",
                            "text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# =============================================
# TAB 3: CARA PENGGUNAAN
# =============================================
with tab3:
    st.header("Cara Penggunaan Aplikasi")
    
    st.markdown("### 1. Demo Prediksi")
    st.markdown("""
    Tab ini digunakan untuk mencoba prediksi model dengan sample data dari dataset.
    
    **Langkah-langkah:**
    1. Pilih jenis sample pada dropdown:
       - **Random Normal** - Mengambil sample transaksi yang aslinya normal
       - **Random Fraud** - Mengambil sample transaksi yang aslinya fraud
       - **Random Transaksi** - Mengambil sample acak (bisa normal atau fraud)
    2. Klik tombol **Generate Sample**
    3. Lihat hasil prediksi:
       - **FRAUD DETECTED** (merah) - Model mendeteksi transaksi sebagai fraud
       - **NORMAL** (hijau) - Model mendeteksi transaksi sebagai normal
    4. Perhatikan **Probabilitas Fraud** - Tingkat keyakinan model (0-100%)
    5. Bandingkan dengan **Label Asli** untuk melihat apakah prediksi benar
    
    **Mengubah Threshold:**
    - Gunakan slider di sidebar untuk mengubah threshold
    - Threshold rendah (misal 30%) = Lebih sensitif, lebih banyak terdeteksi fraud
    - Threshold tinggi (misal 70%) = Lebih ketat, hanya fraud yang sangat yakin
    """)
    
    st.markdown("---")
    st.markdown("### 2. Upload CSV")
    st.markdown("""
    Tab ini digunakan untuk melakukan prediksi pada banyak transaksi sekaligus.
    
    **Format CSV yang dibutuhkan:**
    - Harus memiliki kolom: Time, Amount, V1 sampai V28
    - Kolom Class opsional (jika ada akan diabaikan)
    
    **Langkah-langkah:**
    1. Siapkan file CSV dengan format yang sesuai
    2. Klik area upload atau drag & drop file
    3. Klik tombol **Jalankan Prediksi**
    4. Lihat hasil: Summary jumlah fraud vs normal, Tabel detail setiap transaksi
    5. Download hasil dengan klik **Download Hasil (CSV)**
    """)
    
    st.markdown("---")
    st.markdown("### 3. Memahami Hasil Prediksi")
    st.markdown("""
    | Komponen | Penjelasan |
    |----------|------------|
    | **Probabilitas Fraud** | Persentase keyakinan model bahwa transaksi adalah fraud (0-100%) |
    | **Threshold** | Batas minimum probabilitas untuk dianggap fraud |
    | **Confidence Level** | Visualisasi progress bar dari probabilitas |
    | **Detail Fitur** | Nilai V1-V28 yang digunakan model untuk prediksi |
    """)
    
    st.markdown("**Logika Prediksi:**")
    st.code("""Jika Probabilitas >= Threshold → FRAUD
Jika Probabilitas < Threshold  → NORMAL""")
    
    st.markdown("""
    **Contoh:**
    - Probabilitas: 65%, Threshold: 45% → FRAUD (65% >= 45%)
    - Probabilitas: 40%, Threshold: 45% → NORMAL (40% < 45%)
    """)
    
    st.markdown("---")
    st.markdown("### 4. Tips Penggunaan")
    st.markdown("""
    - **Untuk demo:** Gunakan tab Demo Prediksi, coba beberapa sample fraud dan normal
    - **Untuk analisis:** Gunakan tab Upload CSV dengan data transaksi
    - **Untuk sensitivitas:** Sesuaikan threshold di sidebar sesuai kebutuhan
    """)
    
    st.markdown("---")
    st.markdown("### 5. Catatan Penting")
    st.markdown("""
    - Data V1-V28 adalah hasil transformasi PCA untuk menjaga privasi
    - Model ini dilatih dengan data dari bank di Eropa
    - Performa model: Precision 94%, Recall 82%, F1-Score 87%
    """)

# =============================================
# TAB 4: TENTANG PROJECT
# =============================================
with tab4:
    st.header("Tentang Project")
    
    st.markdown("""
    ### Overview
    
    Project ini membangun sistem deteksi fraud pada transaksi kartu kredit menggunakan 
    Machine Learning. Model dilatih menggunakan dataset dari Kaggle yang berisi 
    transaksi kartu kredit dari bank di Eropa.
    
    ### Dataset
    
    - **Total transaksi:** 284,807
    - **Fraud:** 492 (0.173%)
    - **Normal:** 284,315 (99.827%)
    - **Class imbalance ratio:** 577:1
    
    ### Metodologi
    
    1. **Exploratory Data Analysis** - Memahami karakteristik data
    2. **Preprocessing** - Scaling fitur Amount dan Time
    3. **Handling Imbalance** - Membandingkan berbagai teknik (SMOTE, Class Weight)
    4. **Model Training** - Logistic Regression, Random Forest, XGBoost
    5. **Evaluation** - Fokus pada Precision, Recall, F1-Score, PR-AUC
    6. **Threshold Tuning** - Optimasi berdasarkan kebutuhan bisnis
    
    ### Model Performance
    
    | Metric | Score |
    |--------|-------|
    | Precision | 94.12% |
    | Recall | 81.63% |
    | F1-Score | 87.43% |
    | Optimal Threshold | 0.45 |
    
    ### Tech Stack
    
    - Python, Pandas, NumPy
    - Scikit-learn, XGBoost, Imbalanced-learn
    - Streamlit (UI)
    
    ### Repository
    
    [GitHub Link - Tambahkan link repository di sini]
    """)

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Fraud Detection System | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)