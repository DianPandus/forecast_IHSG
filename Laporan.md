# Laporan Proyek Predictive Analytics - Dian Pandu Syahfitra

## Project Overview

Indeks Harga Saham Gabungan (IHSG) merupakan tolok ukur utama kinerja pasar saham di Indonesia. Pergerakannya dipengaruhi oleh berbagai faktor ekonomi makro, sentimen pasar, dan kinerja emiten. Kemampuan untuk memprediksi pergerakan IHSG memiliki nilai yang signifikan bagi investor dan analis keuangan dalam membuat keputusan investasi yang lebih terinformasi dan mengelola risiko.

**Mengapa dan Bagaimana Masalah Harus Diselesaikan**
Ketidakpastian dalam pergerakan harga saham menjadi tantangan utama. Proyek ini bertujuan untuk mengatasi masalah ini dengan mengembangkan model _machine learning_ yang dapat memprediksi harga penutupan IHSG. Pendekatan ini dilakukan dengan memanfaatkan data historis IHSG, melakukan pra-pemrosesan data, dan menerapkan algoritma _machine learning_ yang sesuai untuk data deret waktu. Dengan demikian, diharapkan dapat menyediakan alat bantu analisis kuantitatif yang lebih objektif.

**Hasil Riset Terkait atau Referensi**
Prediksi pasar saham menggunakan _machine learning_ telah banyak diteliti. Beberapa pendekatan umum melibatkan model statistik klasik dan model _machine learning_ yang lebih modern.

1.  Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. _PLoS ONE, 13_(3), e0194889.
2.  Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. _European Journal of Operational Research, 270_(2), 654-669.
3.  Atsalakis, G. S., & Valavanis, K. P. (2009). Surveying stock market forecasting techniques – Part II: Soft computing methods. _Expert Systems with Applications, 36_(3), 5932-5941.

---

## Business Understanding

Proses klarifikasi masalah dalam proyek ini adalah untuk mengatasi kesulitan dalam memprediksi nilai Indeks Harga Saham Gabungan (IHSG) di masa mendatang. Investor dan _trader_ membutuhkan panduan yang lebih baik daripada sekadar intuisi untuk membuat keputusan investasi. Proyek ini bertujuan untuk mengembangkan model prediktif sebagai salah satu alat bantu dalam proses pengambilan keputusan tersebut.

### Problem Statements

- Bagaimana cara membangun model _machine learning_ yang mampu memprediksi harga penutupan (Close Price) harian IHSG berdasarkan data historisnya?
- Model _machine learning_ manakah (antara Random Forest dengan _lagged features_ dan LSTM) yang memberikan performa prediksi harga penutupan IHSG yang lebih baik pada dataset yang digunakan?

### Goals

- Mengembangkan model Random Forest Regressor yang menggunakan fitur-fitur historis (_lagged features_) dari harga penutupan dan volume perdagangan IHSG untuk melakukan prediksi.
- Mengembangkan model Long Short-Term Memory (LSTM) yang menggunakan sekuens data historis harga penutupan dan volume perdagangan IHSG untuk melakukan prediksi.
- Mengevaluasi dan membandingkan performa kedua model tersebut menggunakan metrik Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE).

### Solution statements

Untuk mencapai _goals_ di atas, diajukan dua pendekatan solusi (_solution approach_):

1.  **Solusi 1: Random Forest Regressor dengan Lagged Features**
    - Pendekatan ini menggunakan algoritma Random Forest untuk memprediksi harga penutupan IHSG. Fitur input yang digunakan adalah nilai-nilai historis (lag) dari harga penutupan dan volume perdagangan IHSG dari beberapa hari sebelumnya.
    - Kinerja akan diukur dengan MAE dan RMSE pada data uji.
2.  **Solusi 2: Long Short-Term Memory (LSTM) Network**
    - Pendekatan ini menggunakan jaringan LSTM, yang merupakan jenis Recurrent Neural Network (RNN) yang cocok untuk data sekuensial seperti deret waktu harga saham. Model akan dilatih menggunakan sekuens data historis harga penutupan dan volume perdagangan.
    - Kinerja akan diukur dengan MAE dan RMSE pada data uji.

---

## Data Understanding

Dataset yang digunakan adalah data historis Indeks Harga Saham Gabungan (IHSG) selama kurang lebih 5 tahun terakhir, dari file `IHSG_5_Tahun.csv` Setelah proses pembersihan dan pra-pemrosesan:

## Data Understanding

Dataset yang digunakan adalah data historis Indeks Harga Saham Gabungan (IHSG) selama kurang lebih 5 tahun terakhir, dari file `IHSG_5_Tahun.csv` yang Anda sediakan. Setelah proses pembersihan dan pra-pemrosesan:

- **Jumlah Data:** Terdapat **1205 entri (baris data)**.
- **Rentang Waktu:** Data mencakup periode dari **26 Mei 2020 hingga 23 Mei 2025**.
- **Kondisi Data:** Semua kolom yang digunakan untuk pemodelan memiliki tipe data numerik (`float64`) dan **tidak ada _missing values_** setelah tahapan penanganan. Indeks data adalah `DatetimeIndex`.
- **Sumber Data:** Data historis IHSG untuk proyek ini diperoleh dari website **Investing.com**. Data diambil untuk rentang waktu sekitar 5 tahun (dari sekitar Mei 2020 hingga Mei 2025) dan kemudian disimpan dalam format file CSV bernama `IHSG_5_Tahun.csv` untuk digunakan dalam proyek ini. Untuk referensi umum mengenai data historis IHSG di Investing.com, dapat diakses melalui pencarian indeks "Jakarta Stock Exchange Composite Index (JKSE)" pada situs tersebut. Tautan halaman umum untuk indeks komposit Jakarta adalah: [https://www.investing.com/indices/idx-composite](https://www.investing.com/indices/idx-composite) (Pengguna dapat menavigasi ke bagian "Historical Data" pada halaman tersebut untuk melihat contoh data yang tersedia).

Variabel-variabel pada dataset setelah pembersihan dan penggantian nama adalah sebagai berikut:

- **Date (Tanggal):** Tanggal pencatatan data (digunakan sebagai indeks).
- **Close (Terakhir):** Harga penutupan IHSG pada hari tersebut. **Ini adalah variabel target yang akan diprediksi.**
- **Open (Pembukaan):** Harga pembukaan IHSG pada hari tersebut.
- **High (Tertinggi):** Harga tertinggi IHSG yang dicapai pada hari tersebut.
- **Low (Terendah):** Harga terendah IHSG yang dicapai pada hari tersebut.
- **Volume (Vol.):** Jumlah saham yang diperdagangkan pada hari tersebut.
- **Change_Percent (Perubahan%):** Persentase perubahan harga penutupan dibandingkan hari sebelumnya.

Untuk model Random Forest, fitur tambahan berupa _lagged features_ juga dibuat (akan dijelaskan di bagian Data Preparation).

**Exploratory Data Analysis (EDA)**
Visualisasi data dilakukan untuk memahami tren dan pola dalam data IHSG.

- **Plot Harga Penutupan IHSG:** Menunjukkan pergerakan harga 'Close' sepanjang waktu, membantu mengidentifikasi tren jangka panjang, volatilitas, dan potensi adanya titik balik atau anomali.
- **Plot Volume Perdagangan IHSG:** Menggambarkan aktivitas perdagangan harian. Lonjakan volume seringkali bertepatan dengan pergerakan harga yang signifikan.

_Contoh snippet kode untuk visualisasi harga penutupan:_

```python
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Harga Penutupan IHSG')
plt.title('Pergerakan Harga Penutupan IHSG (Setelah Pembersihan)')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')
plt.legend()
plt.show()
```

Dari plot ini, Anda dapat mengamati bagaimana IHSG berfluktuasi selama periode 5 tahun tersebut, termasuk potensi tren naik, turun, atau konsolidasi.

---

## Data Preparation

Tahapan persiapan data dilakukan secara berurutan untuk memastikan data siap digunakan oleh model machine learning.

1.  **Pemuatan Data Awal:** Data dari `IHSG_5_Tahun.csv` dimuat sebagai tipe string untuk memungkinkan pembersihan manual format angka.
    - _Alasan:_ Format angka pada file sumber (misalnya, "7.202,97" untuk angka dan "15,22B" untuk volume) tidak bisa langsung dikenali sebagai numerik oleh pandas.
2.  **Pembersihan Nilai Numerik, Volume, dan Persentase:**
    - Fungsi khusus dibuat untuk menghapus pemisah ribuan (misalnya, '.'), mengganti pemisah desimal (misalnya, ',') dengan '.', dan mengonversi akhiran seperti 'B' (Miliar/Billion menjadi E9), 'M' (Juta/Million menjadi E6), 'K' (Ribu/Kilo menjadi E3) pada kolom volume, serta '%' pada kolom perubahan persentase.
    - _Alasan:_ Mengubah data teks menjadi format numerik (float) yang dapat diproses oleh model.
3.  **Konversi Kolom Tanggal:** Kolom 'Tanggal' dikonversi ke tipe `datetime` dengan format yang sesuai (`%d/%m/%Y`).
    - _Alasan:_ Memungkinkan analisis deret waktu dan penggunaan tanggal sebagai indeks.
4.  **Penggantian Nama Kolom:** Nama kolom dari Bahasa Indonesia diubah ke Bahasa Inggris yang lebih standar (misalnya, 'Terakhir' menjadi 'Close').
    - _Alasan:_ Konsistensi dan kemudahan dalam penulisan kode selanjutnya.
5.  **Pengaturan Indeks dan Pengurutan:** Kolom 'Date' dijadikan indeks DataFrame dan data diurutkan berdasarkan tanggal.
    - _Alasan:_ Krusial untuk analisis deret waktu agar data tersusun secara kronologis.
6.  **Penanganan Missing Values:**
    - Kolom 'Volume' diisi menggunakan metode `ffill` (forward fill) lalu diisi 0 jika masih ada NaN di awal.
    - Kolom harga ('Close', 'Open', 'High', 'Low') diisi menggunakan `ffill`.
    - Baris yang masih memiliki NaN pada kolom harga setelah `ffill` akan dihapus (meskipun pada kasus ini, `ffill` seharusnya sudah menangani).
    - _Alasan:_ Model machine learning umumnya tidak bisa menangani _missing values_. `ffill` adalah strategi umum untuk data deret waktu karena nilai hari ini seringkali mirip dengan kemarin.
7.  **Pembagian Data Global:** Dataset dibagi menjadi set training global (80% atau 964 data) dan set testing global (20% atau 241 data) secara kronologis.
    - _Alasan:_ Untuk memiliki set data tes akhir yang benar-benar terpisah untuk mengevaluasi performa model yang sudah final.
8.  **Persiapan Spesifik untuk Random Forest:**
    - **Pembuatan Lagged Features:** Dibuat fitur baru berdasarkan nilai-nilai 'Close' dan 'Volume' dari `n_lags` hari sebelumnya (dalam kasus ini, `n_lags = 10`). Misalnya, `Close_lag_1` adalah harga penutupan satu hari sebelumnya. Total fitur menjadi 20 untuk Random Forest.
    - _Alasan:_ Memberikan konteks historis sebagai input bagi Random Forest, karena model ini tidak secara inheren memahami urutan waktu seperti RNN.
    - **Penghapusan NaN akibat Lagging:** Baris awal yang mengandung NaN karena proses _shifting_ untuk membuat _lag_ dihapus. Ini menghasilkan 954 data training untuk Random Forest.
    - Pembagian data fitur (X) dan target (y) untuk Random Forest, diselaraskan dengan indeks dari set training dan testing global.
9.  **Persiapan Spesifik untuk LSTM:**
    - **Seleksi Fitur:** Kolom 'Close' dan 'Volume' dari set training dan testing global dipilih.
    - **Scaling Fitur:** Nilai 'Close' dan 'Volume' di-_scale_ ke rentang 0-1 menggunakan `MinMaxScaler`.
    - _Alasan:_ LSTM dan jaringan saraf pada umumnya bekerja lebih baik dan konvergen lebih cepat dengan fitur input yang ternormalisasi.
    - **Pembuatan Sekuens:** Data yang sudah di-_scale_ diubah menjadi sekuens input (data dari `sequence_length = 60` hari sebelumnya) dan target output (harga 'Close' hari berikutnya). Ini menghasilkan 904 sekuens training dan 181 sekuens testing untuk LSTM.
    - _Alasan:_ LSTM memproses data dalam bentuk sekuens untuk mempelajari dependensi temporal.

_Contoh snippet kode untuk pembuatan lagged features (Random Forest):_

```python
df_rf = df.copy()
n_lags = 10
for i in range(1, n_lags + 1):
    df_rf[f'Close_lag_{i}'] = df_rf['Close'].shift(i)
    df_rf[f'Volume_lag_{i}'] = df_rf['Volume'].shift(i)
df_rf.dropna(inplace=True)
features_rf = [col for col in df_rf.columns if 'lag' in col]
X_rf = df_rf[features_rf]
y_rf = df_rf['Close']
```

---

## Modeling

Pada tahap ini, dua model _machine learning_ dikembangkan untuk memprediksi harga penutupan IHSG.

### Solusi 1: Random Forest Regressor dengan Lagged Features

Model Random Forest adalah algoritma _ensemble learning_ yang membangun banyak pohon keputusan (_decision trees_) selama pelatihan dan mengeluarkan prediksi rata-rata (regresi) dari masing-masing pohon.

- **Parameter yang Digunakan:**
  - `n_estimators`: 100 (jumlah pohon)
  - `random_state`: 42 (untuk reproduktifitas)
  - `n_jobs`: -1 (menggunakan semua prosesor yang tersedia)
  - `max_depth`: 10 (kedalaman maksimum setiap pohon)
  - `min_samples_split`: 10 (jumlah sampel minimum yang dibutuhkan untuk membelah node internal)
  - `min_samples_leaf`: 5 (jumlah sampel minimum yang dibutuhkan pada leaf node)
    _(Catatan: Parameter ini adalah konfigurasi awal, untuk hasil optimal dapat dilakukan hyperparameter tuning)._
- **Kelebihan:**
  - Mampu menangani hubungan non-linear antar fitur dan target.
  - Relatif robus terhadap _outliers_ dan tidak memerlukan penskalaan fitur yang rumit jika dibandingkan model berbasis gradien.
  - Dapat memberikan ukuran pentingnya fitur (_feature importance_).
- **Kekurangan:**
  - Bisa menjadi "kotak hitam" (kurang interpretatif dibandingkan model linier sederhana).
  - Memerlukan rekayasa fitur yang cermat (seperti _lagged features_) untuk data deret waktu agar efektif.
  - Bisa _overfitting_ pada data yang _noisy_ jika parameter (seperti `max_depth`) tidak diatur dengan baik.

_Contoh snippet kode untuk inisialisasi dan pelatihan Random Forest:_

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1,
                                 max_depth=10, min_samples_split=10, min_samples_leaf=5)
rf_model.fit(X_train_rf, y_train_rf)
predictions_rf = rf_model.predict(X_test_rf)
```

### Solusi 2: Long Short-Term Memory (LSTM) Network

LSTM adalah jenis khusus dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah dependensi jangka panjang dalam data sekuensial, yang sering ditemui pada RNN standar.

- **Arsitektur yang Digunakan:**
  - Input Layer dengan `input_shape=(60, 2)` (60 timesteps, 2 fitur: Close dan Volume yang sudah di-scale).
  - Layer LSTM pertama dengan 50 unit, `return_sequences=True`.
  - Layer Dropout dengan _rate_ 0.2.
  - Layer LSTM kedua dengan 50 unit, `return_sequences=False`.
  - Layer Dropout dengan _rate_ 0.2.
  - Layer Dense dengan 25 unit.
  - Layer Dense output dengan 1 unit (untuk prediksi harga 'Close').
  - Optimizer: 'adam'.
  - Loss function: 'mean_squared_error'.
  - Dilatih selama 30 epoch dengan `EarlyStopping` (patience=10).
- **Kelebihan:**
  - Sangat efektif dalam menangkap pola dan dependensi jangka panjang dalam data deret waktu.
  - Mampu memodelkan hubungan non-linear yang kompleks.
- **Kekurangan:**
  - Membutuhkan lebih banyak data untuk pelatihan yang efektif dibandingkan beberapa model lain.
  - Komputasi lebih intensif dan waktu pelatihan lebih lama.
  - Lebih kompleks untuk diimplementasikan dan di-_tune_ (banyak _hyperparameter_).

_Contoh snippet kode untuk membangun model LSTM:_

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# sequence_length = 60, n_features_lstm = 2
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, n_features_lstm)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=25))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
# history_lstm = model_lstm.fit(...)
```

---

## Evaluation

Evaluasi model dilakukan untuk mengukur seberapa baik model dapat memprediksi harga penutupan IHSG pada data uji (data yang belum pernah dilihat sebelumnya).

Metrik evaluasi utama yang digunakan adalah:

1.  **Mean Absolute Error (MAE)**
2.  **Root Mean Squared Error (RMSE)**

**Penjelasan Metrik dan Formula:**

- **Mean Absolute Error (MAE)**

  - Formula: $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  - Di mana $n$ adalah jumlah data uji, $y_i$ adalah nilai aktual, dan $\hat{y}_i$ adalah nilai prediksi.
  - **Cara Kerja:** MAE mengukur rata-rata dari selisih absolut antara nilai aktual dan nilai prediksi. MAE memberikan gambaran besarnya error dalam unit yang sama dengan variabel target (poin IHSG). Semakin kecil MAE, semakin akurat prediksi model secara rata-rata.

- **Root Mean Squared Error (RMSE)**
  - Formula: $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
  - **Cara Kerja:** RMSE juga mengukur rata-rata besarnya error, tetapi memberikan bobot yang lebih besar pada error yang lebih besar karena adanya proses pengkuadratan selisih. Seperti MAE, RMSE juga dalam unit yang sama dengan variabel target. Nilai RMSE yang lebih kecil menunjukkan model yang lebih baik. RMSE lebih sensitif terhadap prediksi yang sangat meleset (_outlier_) dibandingkan MAE.

**Hasil Proyek Berdasarkan Metrik Evaluasi (dari output terakhir Anda):**

| Model         | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :------------ | :------------------------ | :----------------------------- |
| Random Forest | **123.65**                | 174.55                         |
| LSTM          | 123.76                    | **152.08**                     |

**Interpretasi Hasil:**

- Berdasarkan **MAE**, model **Random Forest (123.65)** sedikit lebih unggul dibandingkan **LSTM (123.76)**. Keduanya sangat kompetitif, menunjukkan bahwa secara rata-rata selisih absolut prediksi mereka terhadap nilai aktual hampir sama.
- Berdasarkan **RMSE**, model **LSTM (152.08)** secara signifikan lebih unggul dibandingkan **Random Forest (174.55)**. Ini mengindikasikan bahwa LSTM mungkin lebih baik dalam menghindari beberapa kesalahan prediksi yang sangat besar, atau varians kesalahannya secara umum lebih kecil.
- Kedua model menunjukkan performa yang **kompetitif dan sangat baik** sebagai _baseline_. MAE sekitar 123.7 poin IHSG, dengan rata-rata harga IHSG (dari statistik deskriptif data Anda) sekitar 6634, berarti rata-rata error prediksi sekitar **1.86%**. Ini adalah hasil yang sangat menjanjikan dan menunjukkan bahwa kedua pendekatan model berhasil mempelajari pola dari data historis IHSG. Model LSTM dengan RMSE yang lebih rendah mungkin lebih disukai jika menghindari error besar adalah prioritas.

_Contoh snippet kode untuk evaluasi:_

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Untuk Random Forest (asumsi y_test_rf_actual dan predictions_rf sudah ada)
# mae_rf = mean_absolute_error(y_test_rf_actual.values, predictions_rf)
# rmse_rf = np.sqrt(mean_squared_error(y_test_rf_actual.values, predictions_rf))

# Untuk LSTM (asumsi actual_lstm_eval dan pred_lstm_eval sudah ada)
# mae_lstm = mean_absolute_error(actual_lstm_eval, pred_lstm_eval)
# rmse_lstm = np.sqrt(mean_squared_error(actual_lstm_eval, pred_lstm_eval))
```

Visualisasi plot perbandingan antara nilai aktual dan prediksi juga sangat penting untuk melihat bagaimana performa model secara kualitatif sepanjang waktu pada data uji.

![Perbandingan LSTM dan RF](gambar.png)

---
