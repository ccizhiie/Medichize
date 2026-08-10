import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 1. Fungsi Pembersihan Teks Kustom
def clean_id_text(text):
    # Mengubah ke huruf kecil (dilakukan manual untuk konsistensi sebelum stopwords)
    text = text.lower()
    # Menghapus tanda baca menggunakan regex
    text = re.sub(r'[^\w\s]', '', text)

    # Daftar stopwords bahasa Indonesia sederhana (Bisa diekspansi menggunakan Sastrawi)
    stopwords = {'saya', 'aku', 'yang', 'di', 'ke', 'dari', 'itu', 'ini', 'dan', 'atau', 'dengan', 'merasa'}

    # Memfilter kata yang tidak ada di daftar stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords]

    return " ".join(cleaned_words)

def train_mental_health_model():
    print("=== Memulai Proses Pelatihan Model ===")

    # Simulasi Dataset (Dalam kasus nyata, ganti dengan pd.read_csv())
    data = {
        'teks_keluhan': [
            "Saya merasa sangat sedih dan tidak berguna setiap malam.",
            "Jantung saya berdebar cepat, selalu cemas dan panik tanpa alasan.",
            "Kehilangan minat pada hobi, hidup terasa hampa dan gelap.",
            "Gelisah terus menerus, takut memikirkan masa depan.",
            "Menangis setiap hari dan ingin menyerah saja.",
            "Tiba-tiba sesak napas dan keringat dingin saat di keramaian."
        ],
        'label': [
            "Depresi", "Kecemasan", "Depresi", "Kecemasan", "Depresi", "Kecemasan"
        ]
    }
    df = pd.DataFrame(data)

    # 2. Membagi data menjadi 80% Training dan 20% Testing
    # stratify=df['label'] memastikan proporsi kelas (Depresi/Kecemasan) seimbang di train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['teks_keluhan'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # 3. Membuat Pipeline: TfidfVectorizer (dengan preprocessor) + LinearSVC
    # Pipeline mencegah kebocoran data (data leakage) dan mempermudah deployment
    model_pipeline = make_pipeline(
        TfidfVectorizer(preprocessor=clean_id_text),
        LinearSVC(random_state=42, dual="auto")
    )

    # Melatih model
    print("Sedang melatih LinearSVC...\n")
    model_pipeline.fit(X_train, y_train)

    # 4. Evaluasi Model: Classification Report
    y_pred = model_pipeline.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 5. Menyimpan Pipeline
    joblib.dump(model_pipeline, 'mental_text_model.pkl')
    print("Pipeline model berhasil disimpan sebagai 'mental_text_model.pkl'")

if __name__ == "__main__":
    train_mental_health_model()
