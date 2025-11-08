# Identifikasi Teks pada Gambar (OCR)

Program Python untuk mendeteksi dan mengenali teks pada gambar menggunakan OpenCV (preprocessing) dan Tesseract OCR (pytesseract).

## Fitur
- Preprocessing: grayscale, thresholding (Otsu), morfologi (open & close) untuk reduksi noise
- OCR: ekstraksi teks dengan pytesseract
- Bounding box pada area teks (pytesseract.image_to_data)
- Tampilkan hasil teks di terminal
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

4. Masukkan nama file gambar (contoh: `nota.jpg` atau `data/teks.png`).
5. Hasil akan ditampilkan di terminal dan disimpan sebagai:
   - `hasil_teks.txt`
   - `hasil_deteksi.jpg`

## Tips Gambar
- Gambar dengan kontras teks yang baik akan menghasilkan OCR lebih akurat.
- Jika teks kecil/blur, coba perbesar resolusi atau gunakan pencahayaan lebih baik.

## Lisensi
Proyek tugas kuliah. Gunakan sesuai kebutuhan pembelajaran.
