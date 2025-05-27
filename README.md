# 🔍 Sistem Klasifikasi Gambar Cerdas

Sistem ini dirancang untuk menghasilkan prediksi gambar yang **akurat**, **robust**, dan **tidak mudah overfitting**, dengan kombinasi **feature extraction modern** dan **algoritma machine learning tradisional**.

---

## 🔢 Algoritma yang Tersedia

### 1. CNN Pre-trained Feature Extractor + ML Tradisional

* **VGG16** + Random Forest / SVM / KNN
* **ResNet50** + Random Forest / SVM / KNN
* **MobileNetV2** + Random Forest / SVM / KNN

### 2. Computer Vision Tradisional + ML

* Color Histogram + Texture Features + ML
* Edge Detection + Local Binary Pattern + ML

---

## 🎯 Keunggulan Sistem

### 1. ✅ Anti-Overfitting

* Menggunakan **cross-validation** untuk evaluasi yang jujur
* Mengetes banyak kombinasi algoritma, dan memilih yang terbaik otomatis
* Algoritma **ML tradisional lebih stabil** di dataset kecil

### 2. 🧠 Smart Feature Extraction

* Menggunakan **model pre-trained** seperti VGG16, ResNet50 yang telah dilatih pada jutaan gambar
* Fitur **tradisional** seperti histogram warna, edge detection, dan tekstur
* **Feature scaling otomatis** untuk memastikan distribusi yang tepat

### 3. 🔒 Robust Prediction

* Menggunakan **confidence threshold** ketat (min. 60%)
* **Perbandingan antar model** sebelum memilih yang terbaik
* Generalisasi lebih baik untuk gambar baru

### 4. ⚙️ Automatic Model Selection

* Semua kombinasi algoritma diuji secara otomatis
* Model terbaik dipilih berdasarkan hasil cross-validation
* **Model terbaik disimpan (persistence)** untuk digunakan kembali

---

## 🚀 Alur Kerja Sistem

1. **Feature Extraction Phase**

   * Menguji fitur dari VGG16 vs fitur tradisional

2. **Algorithm Testing**

   * Menguji Random Forest, SVM, dan KNN pada setiap fitur

3. **Cross-Validation**

   * Mengevaluasi performa secara objektif

4. **Best Model Selection**

   * Memilih kombinasi feature + ML terbaik

5. **Deployment**

   * Model terbaik siap digunakan dalam API produksi

---

## 💡 Mengapa Sistem Ini Lebih Baik?

* ✅ Lebih tahan terhadap **overfitting** dengan ML tradisional dan validasi ketat
* ✅ Representasi fitur lebih baik menggunakan **CNN pre-trained**
* ✅ Penanganan otomatis untuk **hyperparameter**
* ✅ Confidence score lebih jujur
* ✅ Cocok untuk **dataset kecil maupun besar**
