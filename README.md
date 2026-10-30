# Prediksi Performa Akademik Siswa Menggunakan Machine Learning

Repositori ini berisi implementasi tugas project mata kuliah **Kecerdasan Artifisial**. Project ini berfokus pada prediksi capaian nilai akademik siswa (`GradeClass`) menggunakan pendekatan komparatif antara algoritma probabilistik (**Gaussian Naive Bayes**) dan ensemble learning (**Random Forest Classifier**), dengan penanganan *class imbalance* menggunakan **SMOTE** dan pembanding **SMOTENC**.

**Dataset:** *Students Performance Dataset* (El Kharoua, 2024), DOI `10.34740/KAGGLE/DS/5195702`.

> Dataset ini bersifat sintetis dan disediakan untuk keperluan edukasi. Hasil model bersifat *proof-of-concept* dan tidak dapat digeneralisasi langsung ke populasi siswa nyata.

---

## Struktur Direktori

```text
AI/
│
├── dataset/
│   └── Student_performance_data _.csv   # Dataset performa siswa (2.392 data)
│
├── output/                              # Grafik hasil visualisasi EDA & evaluasi
│   ├── 01_eda.png
│   ├── 02_heatmap_korelasi.png
│   ├── 03_smote_comparison.png
│   ├── 04_confusion_matrix.png
│   ├── 05_perbandingan_metrik.png
│   ├── 06_feature_importance.png
│   ├── 07_ablation_smote.png
│   ├── 08_cv_comparison.png
│   └── 09_ablation_smotenc.png
│
├── student_performance.ipynb            # Jupyter Notebook eksperimen lengkap
├── requirements.txt                     # Daftar dependency library Python
└── README.md                            # Dokumentasi ringkas project
```

---

## Ringkasan Dataset

- **Sumber Data:** *Students Performance Dataset* (El Kharoua, 2024), Kaggle.
- **Jumlah Sampel:** 2.392 baris data siswa, 15 kolom.
- **Fitur Aktif:** 12 fitur setelah `StudentID` dan `GPA` dihapus. `GPA` dihapus karena merupakan sumber *data leakage* terhadap `GradeClass`.
- **Target Variabel:** `GradeClass` (5 kelas):
  - `0`: Grade A (3,50–4,00) — 107 siswa (4,5%)
  - `1`: Grade B (3,00–3,49) — 269 siswa (11,2%)
  - `2`: Grade C (2,50–2,99) — 391 siswa (16,3%)
  - `3`: Grade D (2,00–2,49) — 414 siswa (17,3%)
  - `4`: Grade F (< 2,00) — 1.211 siswa (50,6%)
- **Kualitas Data:** Tidak ada *missing value* maupun data duplikat.

---

## Alur Kerja & Metodologi

1. **Exploratory Data Analysis (EDA):** statistik deskriptif, distribusi target, korelasi antarfitur, dan deteksi *class imbalance*.
2. **Pre-processing:**
   - Menghapus `StudentID` dan `GPA` untuk menghindari *data leakage*.
   - Pembagian data 80% latih (1.913 data) dan 20% uji (479 data) dengan *stratified split*.
   - Standardisasi menggunakan `StandardScaler` (fit hanya pada data latih).
   - Penyeimbangan data latih menggunakan **SMOTE**; data uji tetap tidak seimbang agar evaluasi mencerminkan kondisi nyata.
   - Sebagai pembanding metodologis, **SMOTENC** diterapkan dengan `StandardScaler` hanya pada fitur numerik dan fitur kategorikal dilewatkan tanpa scaling.
3. **Pemodelan:**
   - **Gaussian Naive Bayes** sebagai *baseline* probabilistik.
   - **Random Forest Classifier** dengan 100 pohon keputusan.
4. **Evaluasi:**
   - Accuracy, precision, recall, F1-score, macro-F1, recall kelas F, dan recall kelas A.
   - *Confusion matrix* dan *feature importance*.
   - *Baseline majority* (`DummyClassifier`).
   - *Ablation study*: tanpa SMOTE vs SMOTE vs SMOTENC.
   - Validasi silang 5-fold dengan pipeline agar scaler dan resampling hanya diterapkan pada data latih setiap fold.

---

## Ringkasan Hasil Eksperimen

### Evaluasi pada split 80/20 (`random_state=42`)

| Skenario | Model | Accuracy | Macro-F1 | Recall F | Recall A |
|---|---|---:|---:|---:|---:|
| Tanpa SMOTE | Gaussian Naive Bayes | 67,22% | 0,443 | 0,926 | 0,048 |
| Tanpa SMOTE | Random Forest | **72,23%** | **0,557** | 0,914 | 0,190 |
| SMOTE | Gaussian Naive Bayes | 61,59% | 0,460 | 0,798 | 0,238 |
| SMOTE | Random Forest | 68,89% | 0,525 | 0,885 | 0,190 |
| SMOTENC | Gaussian Naive Bayes | 55,32% | 0,396 | 0,765 | 0,286 |
| SMOTENC | Random Forest | 67,01% | 0,522 | 0,840 | 0,238 |

### Validasi silang 5-fold (mean)

| Model | Skenario | Accuracy | Macro-F1 | Recall F |
|---|---|---:|---:|---:|
| Gaussian Naive Bayes | Tanpa SMOTE | 66,0% | 0,441 | 0,912 |
| Gaussian Naive Bayes | SMOTE | 61,0% | 0,471 | 0,794 |
| Gaussian Naive Bayes | SMOTENC | 53,0% | 0,393 | 0,738 |
| Random Forest | Tanpa SMOTE | 71,4% | 0,545 | 0,929 |
| Random Forest | SMOTE | 69,6% | 0,543 | 0,902 |
| Random Forest | SMOTENC | 67,6% | 0,523 | 0,881 |

### Temuan Utama

1. **Random Forest adalah model terbaik** dari sisi akurasi dan macro-F1 pada semua skenario resampling.
2. Untuk konteks *early warning* siswa berisiko gagal (kelas F), model **tanpa resampling** memberikan recall F tertinggi.
3. **SMOTE bukan perbaikan tanpa biaya.** SMOTE menaikkan recall kelas A pada Naive Bayes (4,8% menjadi 23,8%), tetapi menurunkan akurasi dan recall kelas F (92,6% menjadi 79,8%).
4. **SMOTENC tidak meningkatkan akurasi atau macro-F1.** Dibandingkan SMOTE, recall A SMOTENC lebih tinggi (Naive Bayes 28,6%; Random Forest 23,8%), tetapi recall F turun (Naive Bayes 76,5%; Random Forest 84,0%). Penanganan fitur kategorikal yang lebih tepat menggeser kesalahan ke kelas A, bukan menghilangkannya.
5. Fitur dengan kontribusi prediktif terbesar pada Random Forest dengan SMOTE adalah **Absences** (34,42%), diikuti **StudyTimeWeekly** (12,78%), **ParentalSupport** (10,73%), serta **ParentalEducation** dan **Age** (masing-masing sekitar 7,7%). *Feature importance* bersifat prediktif, bukan kausal.
6. **Keterbatasan:** dataset sintetis, hyperparameter belum dioptimalkan, performa kelas A/B masih rendah, dan hasil utama berasal dari satu split 80/20 meskipun sudah diperiksa dengan validasi silang. SMOTE dan SMOTENC telah diuji, tetapi bukan solusi otomatis untuk ketidakseimbangan kelas.

---

## Menjalankan Project

### 1. Prasyarat

Gunakan Python 3.9+ dan Jupyter Notebook/VS Code.

### 2. Instalasi Library

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Eksperimen

Buka dan jalankan notebook dari root project:

```bash
jupyter notebook student_performance.ipynb
```

Seluruh gambar visualisasi akan otomatis disimpan ke folder `output/`. Notebook dapat juga dibuka dan dijalankan langsung melalui VS Code.


