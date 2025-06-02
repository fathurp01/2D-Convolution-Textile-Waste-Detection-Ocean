


          
Saya perlu melihat struktur dan konten file-file dalam folder Tubes untuk memahami proyek ini sebelum membuat README yang informatif.
        
            
toolName: view_files
            
status: success
          
            
filePath: d:\Src\Latihan\PCD\Tubes\main.py
d:\Src\Latihan\PCD\Tubes\image_processing.py
d:\Src\Latihan\PCD\Tubes\haar_processing.py
          
Sekarang saya perlu melihat lebih banyak konten dari file-file utama untuk memahami fungsi lengkap proyek ini.
        
            
toolName: view_files
            
status: success
          
            
filePath: d:\Src\Latihan\PCD\Tubes\main.py
d:\Src\Latihan\PCD\Tubes\image_processing.py
d:\Src\Latihan\PCD\Tubes\haar_processing.py
          
Saya perlu melihat bagian akhir dari file-file ini untuk memahami fungsi lengkap, terutama fungsi utama detect_sampah.
        
            
toolName: view_files
            
status: success
          
            
filePath: d:\Src\Latihan\PCD\Tubes\haar_processing.py
d:\Src\Latihan\PCD\Tubes\laporan.txt
          
Saya perlu melihat struktur folder dan file lainnya untuk memahami komponen lengkap proyek ini.
        
            
toolName: view_folder
            
status: success
          
d:\Src\Latihan\PCD\Tubes
          
Saya perlu melihat file-file dalam folder train_haar_cascade untuk memahami proses training yang digunakan.
        
            
toolName: view_files
            
status: success
          
            
filePath: d:\Src\Latihan\PCD\Tubes\train_haar_cascade\create_annotations.py
d:\Src\Latihan\PCD\Tubes\train_haar_cascade\train_cascade.py
d:\Src\Latihan\PCD\Tubes\dataset\note.txt
          
# Textile Waste Detection System

Sistem deteksi sampah tekstil menggunakan teknik pengolahan citra digital dan Haar Cascade Classifier untuk mengidentifikasi dan memisahkan sampah tekstil dari sampah jenis lainnya di lingkungan laut.

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi sistem deteksi sampah tekstil yang bertujuan untuk membantu proses pemilahan sampah di perairan laut. Sistem ini menggunakan pendekatan pengolahan citra tradisional dengan metode konvolusi 2D dan Haar Cascade Classifier, tanpa mengandalkan deep learning, sehingga dapat menjadi alternatif praktis untuk pengelolaan sampah laut.

### Latar Belakang
Sampah tekstil di perairan laut menjadi masalah serius yang mempengaruhi ekosistem pesisir dan biota laut. Sistem ini dikembangkan untuk membantu proses identifikasi dan pemilahan sampah tekstil secara otomatis menggunakan teknologi pengolahan citra digital.

## 🚀 Fitur Utama

- **Deteksi Otomatis**: Mendeteksi sampah tekstil dalam gambar menggunakan Haar Cascade
- **Preprocessing Adaptif**: Analisis kualitas gambar otomatis dan preprocessing yang disesuaikan
- **GUI Interaktif**: Antarmuka pengguna yang mudah digunakan dengan PyQt5
- **Multi-scale Detection**: Deteksi dengan berbagai parameter untuk akurasi optimal
- **Visualisasi Hasil**: Tampilan hasil deteksi dengan confidence score
- **Batch Processing**: Kemampuan memproses multiple gambar
- **Export Results**: Menyimpan hasil deteksi dan preprocessing

## 🛠️ Teknologi yang Digunakan

- **Python 3.9+**
- **OpenCV**: Untuk pengolahan citra dan deteksi objek
- **PyQt5**: Untuk antarmuka pengguna grafis
- **NumPy**: Untuk operasi array dan matematika
- **Matplotlib**: Untuk visualisasi data

## 📁 Struktur Proyek

```
Tubes/
├── main.py                    # File utama aplikasi GUI
├── image_processing.py        # Modul preprocessing citra
├── haar_processing.py         # Modul deteksi Haar Cascade
├── laporan.txt               # Laporan lengkap proyek
├── dataset/                  # Dataset untuk training
│   ├── positives/           # Gambar sampel positif
│   ├── negatives/           # Gambar sampel negatif
│   ├── info.lst            # File info sampel positif
│   ├── bg.txt              # File daftar background
│   └── note.txt            # Catatan dataset
├── haarcascade_sampah/       # Model Haar Cascade terlatih
├── train_haar_cascade/       # Script untuk training model
│   ├── create_annotations.py # Tool untuk membuat anotasi
│   ├── train_cascade.py     # Script training cascade
│   └── test_cascade.py      # Script testing model
├── output/                   # Hasil preprocessing dan deteksi
├── samples/                  # Sampel gambar untuk testing
└── test_images/             # Gambar uji coba
```

## 🔧 Instalasi

### Persyaratan Sistem
- Python 3.9 atau lebih baru
- OpenCV dengan dukungan Haar Cascade
- PyQt5
- NumPy
- Matplotlib

### Langkah Instalasi

1. **Clone atau download proyek ini**
```bash
cd d:\Src\Latihan\PCD\Tubes
```

2. **Install dependencies**
```bash
pip install opencv-python PyQt5 numpy matplotlib
```

3. **Verifikasi instalasi OpenCV**
```bash
python -c "import cv2; print(cv2.__version__)"
```

## 🎯 Cara Penggunaan

### Menjalankan Aplikasi GUI

```bash
python main.py
```

### Fitur-fitur Aplikasi

1. **Load Image**: Pilih gambar yang akan dianalisis
2. **Processing Options**:
   - Show all preprocessing steps
   - Use adaptive preprocessing
   - Save intermediate results
3. **Detection Settings**: Konfigurasi parameter deteksi
4. **Start Processing**: Mulai proses deteksi
5. **View Results**: Lihat hasil deteksi dengan confidence score

### Penggunaan Programmatic

```python
from haar_processing import detect_sampah
from image_processing import show_and_save_all_processes

# Preprocessing gambar
show_and_save_all_processes('path/to/image.jpg')

# Deteksi sampah tekstil
detections_count, confidences = detect_sampah(
    'path/to/image.jpg', 
    use_preprocessing=True
)

print(f"Ditemukan {detections_count} objek sampah tekstil")
```

## 🧠 Metodologi

### 1. Preprocessing Adaptif
- **Analisis Kualitas Gambar**: Brightness, contrast, noise, blur analysis
- **Grayscale Conversion**: Konversi RGB ke grayscale dengan weighted method
- **Histogram Equalization**: CLAHE untuk peningkatan kontras lokal
- **Filtering**: Bilateral filter untuk noise reduction
- **Sharpening**: Adaptive sharpening berdasarkan blur level
- **Edge Detection**: Enhanced Sobel edge detection

### 2. Deteksi Haar Cascade
- **Multi-scale Detection**: Deteksi dengan berbagai parameter
- **Confidence Calculation**: Perhitungan confidence berdasarkan size, position, aspect ratio
- **Non-Maximum Suppression**: Eliminasi deteksi yang overlap
- **Enhanced Visualization**: Visualisasi hasil dengan color coding

### 3. Training Custom Cascade
- **Dataset Preparation**: Positive dan negative samples
- **Annotation Tool**: Tool untuk membuat bounding box annotations
- **Training Process**: OpenCV cascade training dengan parameter optimal

## 📊 Hasil dan Evaluasi

Sistem ini telah diuji dengan berbagai kondisi gambar:
- ✅ Gambar dengan pencahayaan normal
- ✅ Gambar dengan pencahayaan rendah
- ✅ Gambar dengan noise
- ✅ Gambar dengan blur
- ✅ Gambar dengan kontras rendah

### Metrik Evaluasi
- **Detection Count**: Jumlah objek yang terdeteksi
- **Confidence Score**: Tingkat kepercayaan deteksi (10-100%)
- **Processing Time**: Waktu pemrosesan per gambar
- **Accuracy**: Akurasi deteksi berdasarkan ground truth

## 🔄 Training Model Baru

Untuk melatih model Haar Cascade dengan dataset sendiri:

1. **Siapkan Dataset**
```bash
# Letakkan gambar positif di dataset/positives/
# Letakkan gambar negatif di dataset/negatives/
```

2. **Buat Annotations**
```bash
python train_haar_cascade/create_annotations.py
```

3. **Train Cascade**
```bash
python train_haar_cascade/train_cascade.py --num_stages 20
```

## 🐛 Troubleshooting

### Error: "Haar cascade file not found"
- Pastikan file `haarcascade_sampah/cascade.xml` ada
- Atau gunakan Browse Cascade untuk memilih file cascade lain

### Error: "Cannot load image"
- Pastikan format gambar didukung (JPG, PNG, BMP)
- Periksa path gambar sudah benar

### Deteksi tidak akurat
- Coba aktifkan "Use adaptive preprocessing"
- Sesuaikan parameter deteksi di Advanced Settings
- Pastikan gambar memiliki kualitas yang baik

## 📈 Pengembangan Selanjutnya

- [ ] Implementasi deep learning untuk akurasi lebih tinggi
- [ ] Support untuk video processing
- [ ] Real-time detection menggunakan webcam
- [ ] Mobile app implementation
- [ ] Cloud-based processing
- [ ] Integration dengan sistem monitoring lingkungan

## 📚 Referensi

1. Wikiandy, Rosidah, dan Titin Herawati (2013). "Dampak Pencemaran Limbah Industri Tekstil Terhadap Kerusakan Struktur Organ Ikan di DAS Citarum Bagian Hulu"
2. OpenCV Documentation - Haar Cascade Classifiers
3. Digital Image Processing Techniques

## 👥 Kontributor

Proyek ini dikembangkan sebagai tugas akhir mata kuliah Pengolahan Citra Digital (PCD).

## 📄 Lisensi

Proyek ini dibuat oleh Fathurrahman Pratama Putra untuk keperluan akademik dan penelitian.

---

        