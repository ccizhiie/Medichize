import pandas as pd
import numpy as np

def run_eda():
    print("=== Memulai Exploratory Data Analysis ===")

    # Simulasi memuat dataset Kaggle (Ganti dengan pd.read_csv('mental_health.csv'))
    np.random.seed(42)
    df_mental = pd.DataFrame({
        'Marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 1000),
        'Depression': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7])
    })

    # 1. Persentase hubungan Marital_status dengan Depression
    print("\n[1] Persentase Depresi berdasarkan Status Pernikahan:")
    depression_rates = df_mental.groupby('Marital_status')['Depression'].value_counts(normalize=True).unstack() * 100
    print(depression_rates)

    # Simulasi memuat dataset Gejala (Ganti dengan pd.read_csv('symptoms.csv'))
    df_symptoms = pd.DataFrame({
        'Patient_ID': [1, 2, 3],
        'Fever': [1, 0, 1],
        'Cough': [0, 1, 1],
        'Fatigue': [1, 1, 0]
    })

    # 2. Merombak format data gejala medis (Wide ke Long format) menggunakan melt
    print("\n[2] Merombak data gejala (Wide to Long) untuk keperluan visualisasi:")
    df_symptoms_long = pd.melt(
        df_symptoms,
        id_vars=['Patient_ID'],
        value_vars=['Fever', 'Cough', 'Fatigue'],
        var_name='Symptom',
        value_name='Is_Present'
    )

    # Filter hanya gejala yang dialami (Is_Present == 1)
    df_symptoms_long = df_symptoms_long[df_symptoms_long['Is_Present'] == 1]
    print(df_symptoms_long.head())

if __name__ == "__main__":
    run_eda()
