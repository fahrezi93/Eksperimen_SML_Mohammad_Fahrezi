import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def run_preprocessing():
    print("Memulai proses Data Preprocessing...")
    
    # 1. Mendefinisikan Path
    raw_data_path = '../data_raw/insurance.csv'
    clean_data_dir = '../preprocessing/data_preprocessing'
    clean_data_path = f'{clean_data_dir}/insurance_clean.csv'
    
    # Memastikan direktori tujuan ada
    os.makedirs(clean_data_dir, exist_ok=True)
    
    # 2. Load Data
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Data raw berhasil dimuat dari {raw_data_path} dengan ukuran {df.shape}")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {raw_data_path}")
        return
        
    df_clean = df.copy()

    # 3. Encoding Kategori Data
    print("Melalukan Encoding pada kolom kategorikal...")
    label_encoder = LabelEncoder()
    df_clean['sex'] = label_encoder.fit_transform(df_clean['sex'])
    df_clean['smoker'] = label_encoder.fit_transform(df_clean['smoker'])
    df_clean['region'] = label_encoder.fit_transform(df_clean['region'])

    # 4. Standarisasi Data Numerik (Scaling)
    print("Melakukan Standarisasi pada kolom numerik...")
    scaler = StandardScaler()
    num_cols = ['age', 'bmi', 'children']
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    # 5. Menyimpan Data Bersih
    df_clean.to_csv(clean_data_path, index=False)
    print(f"Selesai! Data bersih berhasil disimpan di: {clean_data_path}")

if __name__ == "__main__":
    run_preprocessing()
