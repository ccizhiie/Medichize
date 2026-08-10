import streamlit as st
import pandas as pd
import joblib
import re

# Fungsi pembersihan teks kustom (wajib didefinisikan agar joblib dapat memuat model)
def clean_id_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords = {'saya', 'aku', 'yang', 'di', 'ke', 'dari', 'itu', 'ini', 'dan', 'atau', 'dengan', 'merasa'}
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords]
    return " ".join(cleaned_words)

# Memuat model teks NLP dengan caching
@st.cache_resource
def load_text_model():
    try:
        return joblib.load('mental_text_model.pkl')
    except FileNotFoundError:
        return None

# Database lokal fasilitas kesehatan mental nyata di Indonesia
def get_faskes_data():
    data = [
        {
            "nama": "RS Jiwa Dr. Soeharto Heerdjan",
            "kota": "Jakarta",
            "alamat": "Jl. Prof. Dr. Latumeten No.1, Grogol Petamburan",
            "lat": -6.1625,
            "lon": 106.7885
        },
        {
            "nama": "RSUPN Dr. Cipto Mangunkusumo (Poli Psikiatri)",
            "kota": "Jakarta",
            "alamat": "Jl. Pangeran Diponegoro No.71, Senen",
            "lat": -6.2001,
            "lon": 106.8488
        },
        {
            "nama": "RS Menur Surabaya",
            "kota": "Surabaya",
            "alamat": "Jl. Menur No.120, Airlangga, Gubeng",
            "lat": -7.2845,
            "lon": 112.7562
        },
        {
            "nama": "RSUD Dr. Soetomo",
            "kota": "Surabaya",
            "alamat": "Jl. Prof. Dr. Moestopo No.6-8, Airlangga",
            "lat": -7.2693,
            "lon": 112.7554
        },
        {
            "nama": "RSUP Dr. Hasan Sadikin",
            "kota": "Bandung",
            "alamat": "Jl. Pasteur No.38, Pasteur, Cicendo",
            "lat": -6.8987,
            "lon": 107.5973
        },
        {
            "nama": "RS Jiwa Provinsi Jawa Barat",
            "kota": "Bandung",
            "alamat": "Jl. Kolonel Masturi Km. 7, Cisarua, Bandung Barat",
            "lat": -6.8152,
            "lon": 107.5511
        }
    ]
    return pd.DataFrame(data)

def main():
    st.set_page_config(page_title="AI Mental Health Care Hub", layout="wide")

    # Navigasi Sidebar
    st.sidebar.title("Navigasi Menu")
    page = st.sidebar.radio("Pilih Halaman:", ["Psychological Symptom Navigator", "Direktori Faskes & Psikiater"])

    # --- HALAMAN 1: NAVIGATOR & PERTOLONGAN PERTAMA ---
    if page == "Psychological Symptom Navigator":
        st.header("Psychological Symptom Navigator")
        st.write("Analisis pola bahasa untuk mendeteksi kecenderungan kondisi psikologis dan mendapatkan solusi pertolongan pertama mandiri.")

        model_pipeline = load_text_model()

        if not model_pipeline:
            st.error("Model 'mental_text_model.pkl' tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
            return

        # Input teks bebas
        user_story = st.text_area(
            "Ceritakan kendala, masalah percintaan, atau apa yang mengganggu pikiran Anda akhir-akhir ini...",
            height=150,
            placeholder="Ketik keluhan Anda di sini..."
        )

        if st.button("Analisis Masalah Saya"):
            if not user_story.strip():
                st.warning("Silakan tuliskan cerita Anda terlebih dahulu.")
            else:
                # Prediksi menggunakan model ML
                prediksi = model_pipeline.predict([user_story])[0]

                # Logika rekomendasi dan First-Aid Solutions
                if prediksi == "Depresi":
                    spesialis = "Psikolog Klinis atau Psikiater (untuk evaluasi gejala afektif)"
                    first_aid = [
                        "**Teknik Behavioral Activation:** Lakukan satu aktivitas kecil yang dulu Anda sukai (misalnya berjalan kaki 10 menit atau merapikan meja) untuk memecah siklus isolasi.",
                        "**Teknik Self-Compassion:** Akui bahwa perasaan berat ini valid. Hindari menyalahkan diri sendiri dan tuliskan 3 hal kecil yang patut disyukuri hari ini."
                    ]
                elif prediksi == "Kecemasan":
                    spesialis = "Psikolog Klinis (untuk terapi kognitif/CBT terkait kepanikan)"
                    first_aid = [
                        "**Teknik Pernapasan 4-7-8:** Tarik napas selama 4 detik, tahan selama 7 detik, dan hembuskan perlahan selama 8 detik untuk menstabilkan sistem saraf otonom.",
                        "**Teknik Grounding 5-4-3-2-1:** Sebutkan 5 hal yang bisa dilihat, 4 hal yang bisa disentuh, 3 hal yang didengar, 2 hal yang dicium, dan 1 hal yang dirasakan untuk mengalihkan kepanikan."
                    ]
                else:
                    spesialis = "Tenaga Kesehatan Mental Profesional"
                    first_aid = [
                        "**Teknik Pernapasan Diafragma:** Ambil napas dalam melalui hidung, kembangkan perut, lalu hembuskan perlahan.",
                        "**Jurnaling Emosi:** Tuliskan seluruh uneg-uneg secara bebas di atas kertas untuk mengurangi beban kognitif."
                    ]

                # Menampilkan Hasil Analisis
                st.success(f"**Hasil Analisis Berdasarkan Pola Bahasa Anda:** Terindikasi {prediksi}")
                st.info(f"**Rekomendasi Rujukan:** Silakan konsultasikan dengan {spesialis}")

                # Menampilkan Solusi Pertolongan Pertama (First-Aid Solutions)
                st.markdown("### 💡 Solusi Pertolongan Pertama (First-Aid Solutions)")
                for aid in first_aid:
                    st.markdown(f"- {aid}")

        st.markdown("---")
        st.caption("⚠️ Disclaimer: Sistem AI ini hanya mendeteksi pola bahasa dan bukan merupakan diagnosis medis konklusif.")

    # --- HALAMAN 2: DIREKTORI FASKES & PETA INTERAKTIF ---
    elif page == "Direktori Faskes & Psikiater":
        st.header("Direktori Faskes & Psikiater Terdekat")
        st.write("Temukan rumah sakit atau klinik kesehatan jiwa rujukan resmi di kota-kota besar Indonesia.")

        df_faskes = get_faskes_data()

        # Layout Filter menggunakan st.columns
        col1, col2 = st.columns(2)
        with col1:
            selected_city = st.selectbox("Filter berdasarkan Kota:", ["Semua Kota"] + list(df_faskes['kota'].unique()))

        # Filter data faskes berdasarkan pilihan kota
        if selected_city != "Semua Kota":
            filtered_df = df_faskes[df_faskes['kota'] == selected_city]
        else:
            filtered_df = df_faskes

        st.markdown(f"Menampilkan **{len(filtered_df)}** fasilitas kesehatan:")

        # Menampilkan daftar faskes dalam bentuk expander interaktif
        for index, row in filtered_df.iterrows():
            with st.expander(f"🏥 {row['nama']} ({row['kota']})"):
                st.write(f"**Alamat:** {row['alamat']}")
                st.write(f"**Kota:** {row['kota']}")
                st.write(f"**Koordinat:** Latitude {row['lat']}, Longitude {row['lon']}")

        st.markdown("---")
        st.subheader("📍 Visualisasi Peta Persebaran Faskes")
        # Visualisasi peta interaktif Streamlit
        st.map(filtered_df[['lat', 'lon']])

if __name__ == "__main__":
    main()
