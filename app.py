import streamlit as st
import pandas as pd
import joblib
import re
import math

# --- LOGIKA BACKEND ---

def clean_id_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords = {'saya', 'aku', 'yang', 'di', 'ke', 'dari', 'itu', 'ini', 'dan', 'atau', 'dengan', 'merasa'}
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords]
    return " ".join(cleaned_words)

@st.cache_resource
def load_text_model():
    try:
        return joblib.load('mental_text_model.pkl')
    except FileNotFoundError:
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Jari-jari bumi dalam kilometer
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_faskes_data():
    data = [
        {"nama": "RSUD Kanjuruhan", "kota": "Malang", "alamat": "Jl. Panji No.100, Penarukan, Kepanjen", "lat": -8.1353, "lon": 112.5738},
        {"nama": "RS Jiwa Dr. Radjiman Wediodiningrat", "kota": "Malang", "alamat": "Jl. Jend. Ahmad Yani, Lawang", "lat": -7.8340, "lon": 112.6980},
        {"nama": "RS Menur Surabaya", "kota": "Surabaya", "alamat": "Jl. Menur No.120, Airlangga, Gubeng", "lat": -7.2845, "lon": 112.7562},
        {"nama": "RS Jiwa Dr. Soeharto Heerdjan", "kota": "Jakarta", "alamat": "Jl. Prof. Dr. Latumeten No.1, Grogol", "lat": -6.1625, "lon": 106.7885},
        {"nama": "RSUP Dr. Hasan Sadikin", "kota": "Bandung", "alamat": "Jl. Pasteur No.38, Pasteur", "lat": -6.8987, "lon": 107.5973}
    ]
    return pd.DataFrame(data)

# --- UI & TEMA ---

def inject_custom_css():
    st.markdown("""
    <style>
        /* Palet Warna & Font */
        .stApp { background-color: #FAFAFA; color: #333333; }

        /* Tombol Estetik */
        .stButton>button {
            border-radius: 8px !important;
            background-color: #4A7C59 !important;
            color: #FFFFFF !important;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3b6347 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Kartu Visual (Visual Anchor) */
        .info-card {
            background-color: #EAF2EF;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #4A7C59;
            margin-bottom: 20px;
            color: #2D3748;
        }

        /* Tombol Darurat Sidebar */
        .emergency-card {
            background-color: #FFF5F5;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #FC8181;
            margin-bottom: 25px;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Mental Health Care Hub", layout="wide")
    inject_custom_css()

    # --- SIDEBAR & TOMBOL DARURAT ---
    with st.sidebar:
        st.markdown("""
        <div class="emergency-card">
            <h4 style="margin-top:0; color: #E53E3E;">🚨 Butuh Bantuan Darurat?</h4>
            <p style="font-size: 13px; color: #4A5568;">Jika Anda merasa dalam krisis, hubungi tenaga profesional sekarang:</p>
            <a href="tel:119" style="text-decoration: none;">
                <div style="background-color: #E53E3E; color: white; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 8px;">📞 Hotline Kemenkes (119)</div>
            </a>
            <a href="https://www.intothelightid.org/tentang-bunuh-diri/hotline-dan-konseling/" target="_blank" style="text-decoration: none;">
                <div style="background-color: #E53E3E; color: white; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold;">🌐 Into The Light ID</div>
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.title("🧭 Navigasi Menu")
        page = st.radio("", ["Psychological Symptom Navigator", "Direktori Faskes & Psikiater"])

    # --- HALAMAN 1: NAVIGATOR & FIRST AID ---
    if page == "Psychological Symptom Navigator":
        st.header("🧠 Psychological Symptom Navigator")
        st.markdown("<p style='color: #4A5568;'>Deteksi pola bahasa klinis dan dapatkan panduan stabilisasi awal.</p>", unsafe_allow_html=True)

        model_pipeline = load_text_model()

        if not model_pipeline:
            st.error("Model 'mental_text_model.pkl' tidak ditemukan. Pastikan Anda telah menjalakan pipeline ML.")
            return

        user_story = st.text_area(
            "Ceritakan apa yang mengganggu pikiran Anda akhir-akhir ini...",
            height=150
        )

        if st.button("🔍 Analisis Masalah Saya"):
            if not user_story.strip():
                st.warning("Silakan tuliskan cerita Anda terlebih dahulu.")
            else:
                prediksi = model_pipeline.predict([user_story])[0]

                # Pemetaan Logika
                if prediksi == "Depresi":
                    spesialis = "Psikolog Klinis / Psikiater"
                    first_aid = [
                        "🎯 **Behavioral Activation:** Lakukan 1 aktivitas kecil yang mudah diselesaikan hari ini (misal: merapikan kasur).",
                        "🫂 **Self-Compassion:** Akui bahwa merasa berat itu valid. Jangan memaksakan diri."
                    ]
                else:
                    spesialis = "Psikolog Klinis"
                    first_aid = [
                        "🌬️ **Pernapasan 4-7-8:** Tarik napas 4 detik, tahan 7 detik, hembuskan 8 detik.",
                        "⚓ **Grounding 5-4-3-2-1:** Temukan 5 hal yang bisa dilihat, 4 disentuh, 3 didengar, 2 dicium, 1 dirasakan."
                    ]

                # Visualisasi Kartu Hasil
                st.markdown(f"""
                <div class="info-card">
                    <h4>📊 Hasil Analisis Pola Bahasa</h4>
                    <p><b>Indikasi Utama:</b> {prediksi}</p>
                    <p><b>Rujukan Profesional:</b> {spesialis}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("💡 Solusi Pertolongan Pertama")
                for aid in first_aid:
                    st.markdown(f"- {aid}")

                # Fitur Eksport Solusi
                export_data = f"HASIL DETEKSI: {prediksi}\nRUJUKAN: {spesialis}\n\nSOLUSI PERTOLONGAN PERTAMA:\n" + "\n".join(first_aid)
                st.download_button(
                    label="📥 Unduh Panduan Ini (.txt)",
                    data=export_data,
                    file_name="first_aid_guide.txt",
                    mime="text/plain"
                )

        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** Analisis NLP ini tidak menggantikan diagnosis medis konklusif.")

    # --- HALAMAN 2: DIREKTORI & GEOSPATIAL TRACKING ---
    elif page == "Direktori Faskes & Psikiater":
        st.header("🏥 Direktori Faskes & Psikiater")
        df_faskes = get_faskes_data()

        # Koordinat jangkar untuk perhitungan jarak
        lokasi_referensi = {
            "Lokasi Saat Ini (Kepanjen, Malang)": (-8.132, 112.572),
            "Pusat Kota Jakarta": (-6.200, 106.816),
            "Pusat Kota Surabaya": (-7.250, 112.750)
        }

        col1, col2 = st.columns(2)
        with col1:
            titik_awal = st.selectbox("📍 Urutkan Berdasarkan Jarak Terdekat Dari:", list(lokasi_referensi.keys()))

        # Eksekusi perhitungan Haversine
        lat_user, lon_user = lokasi_referensi[titik_awal]
        df_faskes['jarak_km'] = df_faskes.apply(
            lambda row: haversine_distance(lat_user, lon_user, row['lat'], row['lon']), axis=1
        )

        # Urutkan berdasarkan jarak terdekat
        df_faskes_sorted = df_faskes.sort_values(by='jarak_km')

        st.markdown(f"Menampilkan fasilitas terdekat dari **{titik_awal}**:")

        for index, row in df_faskes_sorted.iterrows():
            with st.expander(f"🏥 {row['nama']} — {row['jarak_km']:.1f} km"):
                st.write(f"**Kota:** {row['kota']}")
                st.write(f"**Alamat:** {row['alamat']}")

        st.markdown("---")
        st.subheader("🗺️ Peta Persebaran Faskes")
        st.map(df_faskes_sorted[['lat', 'lon']])

if __name__ == "__main__":
    main()
