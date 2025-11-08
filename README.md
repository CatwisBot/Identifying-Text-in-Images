# Identifikasi Teks pada Gambar (OCR) - Enhanced Version

Program Python untuk mendeteksi dan mengenali teks pada gambar menggunakan OpenCV (preprocessing) dan Tesseract OCR (pytesseract) dengan fitur **Enhanced Mode** untuk akurasi maksimal.

## ✨ Fitur Utama

### Mode Enhanced (BARU!)
- **Image Upscaling 2x** - Otomatis memperbesar gambar kecil untuk teks lebih jelas
- **Unsharp Masking** - Mempertajam teks yang blur
- **Contrast Enhancement** - Meningkatkan kontras dokumen
- **Multiple Thresholding** - Otsu, Adaptive, CLAHE (3 strategi berbeda)
- **Multiple PSM Modes** - Mencoba 3 page segmentation modes (PSM 3, 4, 6)
- **Dual Language OCR** - English + Indonesian bersamaan
- **OCR Error Correction** - Koreksi otomatis 40+ kesalahan umum
- **Number Pattern Fixing** - Perbaiki IP address, nomor soal otomatis
- **Debug Output** - Simpan semua preprocessing variants untuk analisis

### Fitur Dasar
- Preprocessing: grayscale, thresholding (Otsu), morfologi (open & close)
- OCR: ekstraksi teks dengan pytesseract
- Bounding box pada area teks
- Tampilkan hasil teks di terminal dengan statistik
- Simpan teks ke `hasil_teks.txt`
- Simpan gambar anotasi ke `hasil_deteksi.jpg`

## Prasyarat
- Python 3.9+ (direkomendasikan)
- Paket Python:
  - opencv-python
  - pytesseract
  - numpy
- Tesseract OCR (aplikasi native, bukan pip) harus terpasang di sistem
  - Windows (umum): `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`

> Catatan: `pytesseract` hanyalah binding ke program Tesseract OCR. Anda tetap harus menginstal Tesseract OCR.

## Instalasi Paket Python
Jalankan perintah di terminal pada folder proyek:

```bash
pip install -r requirements.txt
```

## Instalasi Tesseract OCR (Windows)
- Unduh dan pasang Tesseract OCR untuk Windows (misal build resmi atau dari UB Mannheim).
- Setelah terpasang, pastikan file berikut ada:
  - `C:\\Program Files\\Tesseract-OCR\\tesseract.exe` (atau di `Program Files (x86)`)
- Script akan mencoba mendeteksi path umum itu secara otomatis. Jika Tesseract ada di lokasi lain, Anda bisa mengatur manual di kode, misalnya:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\\Apps\\Tesseract-OCR\\tesseract.exe"
```

Jika ingin OCR Bahasa Indonesia, pastikan paket bahasa `ind.traineddata` terpasang di folder `tessdata` Tesseract.

### Menambahkan Tesseract ke PATH (permanen)
1. Buka: Start Menu -> ketik `Environment Variables` -> pilih "Edit the system environment variables".
2. Klik "Environment Variables...".
3. Pada bagian "System variables" pilih `Path` -> "Edit" -> "New".
4. Tambahkan folder (bukan file) misalnya: `C:\\Program Files\\Tesseract-OCR\\`
5. Klik OK sampai semua dialog tertutup.
6. Buka terminal baru, jalankan:
  ```bash
  tesseract --version
  ```
  Jika versi tampil, PATH sudah benar.

### Jika perintah `python` tidak dikenali
Instal Python dari https://www.python.org/downloads/ dan pastikan mencentang "Add Python to PATH" di awal installer. Verifikasi dengan:
```bash
python --version
pip --version
```
Jika masih gagal, coba gunakan `py -3` di Windows:
```bash
py -3 -m pip install -r requirements.txt
py -3 main.py
```

## Cara Menjalankan
1. Buka folder proyek di VS Code.
2. Pastikan dependency terpasang (lihat bagian Instalasi Paket Python).
3. Jalankan program:

```bash
python main.py
```

4. **Pilih mode preprocessing:**
   - **Mode 1 (Standard)**: Cepat, untuk gambar berkualitas baik
   - **Mode 2 (Enhanced)**: Akurat, untuk dokumen dengan teks kecil/blur - **DIREKOMENDASIKAN**

5. Masukkan nama file gambar (contoh: `soal_ujian.jpg`).
6. Hasil ditampilkan di terminal dan disimpan.

### 🎯 Kapan Menggunakan Mode Enhanced?

**Mode Enhanced sangat direkomendasikan untuk:**
- 📄 Dokumen ujian/soal dengan teks kecil
- 📸 Foto dokumen (bukan scan berkualitas tinggi)
- 🔍 Gambar dengan teks blur atau tidak fokus
- 📱 Screenshot dengan resolusi rendah (<1500px)
- 📊 Dokumen dengan layout kompleks (kolom, tabel)
- 🖨️ Fotokopi yang kualitasnya menurun
- 📝 Tulisan dengan font kecil (<12pt)

**Mode Standard cukup untuk:**
- ✅ Scan dokumen berkualitas tinggi (300+ DPI)
- ✅ Gambar dengan teks besar dan jelas
- ✅ Background putih bersih, teks hitam tebal
- ✅ Kecepatan lebih penting daripada akurasi

### 📊 Hasil yang Diharapkan

**Output di Terminal:**
```
=== HASIL OCR ===
======================================================================
22. Untuk mempermudah pembagian subnet
Untuk merangkum entry routing table
Kelebihan protokol TCP dibanding UDP terutama dari sisi:
a. Beban network lebih kecil
b. Reliabilitas lebih tinggi
...

======================================================================
[*] Total karakter: 1247
[*] Total baris: 45
[*] Total kata: 234
```

**Files yang Dihasilkan:**
- `hasil_teks.txt` - Teks hasil OCR (sudah dikoreksi)
- `hasil_deteksi.jpg` - Gambar dengan bounding boxes
- `debug_otsu.jpg` - Preprocessing Otsu (mode enhanced)
- `debug_adaptive.jpg` - Preprocessing Adaptive (mode enhanced)
- `debug_clahe.jpg` - Preprocessing CLAHE (mode enhanced)
- `debug_enhanced_gray.jpg` - Enhanced grayscale (mode enhanced)

## 🔧 Fitur OCR Error Correction

Program ini secara otomatis mengoreksi kesalahan OCR umum:

### Kesalahan Karakter Umum
- `Unluk` → `Untuk`
- `dan]` → `dari`
- `dani`, `dati` → `dari`
- `tepal` → `tepat`
- `saal` → `saat`
- `vang` → `yang`
- `Iinggi` → `tinggi`
- `Iain` → `lain`
- Dan 30+ koreksi lainnya...

### Pattern Fixing
- **IP Address**: Otomatis ganti huruf O dengan angka 0
  - `192.177.1OO.32` → `192.177.100.32`
- **Nomor Soal**: Perbaiki spacing
  - `22.Untuk` → `22. Untuk`

### Pembersihan Teks
- Hapus karakter form-feed (`\x0c`)
- Normalize whitespace berlebihan
- Trim spasi di awal/akhir baris

## Tips Gambar
- Gambar dengan kontras teks yang baik akan menghasilkan OCR lebih akurat.
- Jika teks kecil/blur, coba perbesar resolusi atau gunakan pencahayaan lebih baik.

## Lisensi
Proyek tugas kuliah. Gunakan sesuai kebutuhan pembelajaran.
