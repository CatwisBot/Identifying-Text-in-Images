"""
Program: Identifikasi Teks pada Gambar (OCR)
Mata Kuliah: Pengolahan Citra Digital

Deskripsi singkat:
- Membaca gambar dari input pengguna
- Preprocessing (grayscale, threshold/binerisasi, morfologi untuk hapus noise)
- OCR menggunakan pytesseract
- Menampilkan hasil teks di terminal
- Menggambar bounding box di sekitar teks terdeteksi
- Menyimpan hasil teks ke "hasil_teks.txt"
- Menyimpan gambar anotasi ke "hasil_deteksi.jpg"

Library: cv2 (OpenCV), pytesseract, numpy, os

Catatan Windows:
Pastikan aplikasi Tesseract OCR sudah terpasang (bukan hanya pip pytesseract).
Installer umum: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
Script ini akan mencoba mendeteksi path tersebut secara otomatis.
"""

from __future__ import annotations

import os
import sys
import shutil
from typing import Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def ensure_tesseract_cmd() -> str | None:
	"""
	Deteksi tesseract.exe dan set untuk pytesseract bila perlu.
	Urutan: coba dari PATH (shutil.which), lalu lokasi umum Windows.
	"""
	# 1) Sudah ada di PATH?
	found = shutil.which("tesseract")
	if found:
		return found

	# 2) Lokasi umum Windows
	if os.name == "nt":
		candidates = [
			r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
			r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
		]
		for p in candidates:
			if os.path.isfile(p):
				pytesseract.pytesseract.tesseract_cmd = p
				return p
	return None


def preprocess_image(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Preprocessing gambar untuk meningkatkan hasil OCR.

	Langkah:
	- Konversi ke grayscale
	- Gaussian blur untuk mengurangi noise frekuensi tinggi
	- Thresholding Otsu (binerisasi)
	- Operasi morfologi (open lalu close) untuk hapus noise kecil dan merapikan kontur teks

	Return:
	- processed: citra hasil akhir (biner) untuk OCR dan deteksi
	- gray: citra grayscale (untuk ditampilkan jika perlu)
	"""
	# 1) Grayscale
	if len(img_bgr.shape) == 3:
		gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	else:
		gray = img_bgr.copy()

	# 2) Kurangi noise (opsional tapi membantu Otsu threshold)
	gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

	# 3) Thresholding Otsu -> menghasilkan citra biner
	_thr, thresh = cv2.threshold(
		gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
	)

	# 4) Operasi morfologi: open (erode -> dilate) untuk hapus noise kecil
	kernel = np.ones((3, 3), np.uint8)
	opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

	# 5) Close (dilate -> erode) untuk menutup celah kecil pada huruf/katakata
	processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

	return processed, gray


def perform_ocr(img_bin: np.ndarray, lang: str = "eng") -> str:
	"""
	Lakukan OCR pada citra biner dengan konfigurasi standar yang cocok untuk blok teks.
	- lang: bahasa Tesseract (default 'eng'). Dapat diubah jika paket bahasa lain terpasang.
	"""
	# --oem 3: LSTM-based engine, --psm 6: Assume a single uniform block of text
	config = "--oem 3 --psm 6"
	text = pytesseract.image_to_string(img_bin, lang=lang, config=config)
	# Bersihkan karakter form-feed yang sering muncul di akhir
	return text.replace("\x0c", "").strip()


def draw_bounding_boxes(
	img_bgr: np.ndarray,
	img_for_detection: np.ndarray,
	lang: str = "eng",
	conf_threshold: float = 60.0,
) -> Tuple[np.ndarray, int]:
	"""
	Menggunakan image_to_data untuk mendapatkan koordinat dan menggambar kotak di sekitar teks.
	- img_for_detection sebaiknya hasil preprocessing (biner) agar deteksi lebih stabil.
	- conf_threshold: ambang minimal confidence (0-100) agar kotak digambar.

	Return: (gambar_beranotasi, jumlah_kotak)
	"""
	config = "--oem 3 --psm 6"
	data = pytesseract.image_to_data(
		img_for_detection, lang=lang, config=config, output_type=Output.DICT
	)

	annotated = img_bgr.copy()
	n = len(data["level"]) if "level" in data else 0
	count = 0

	for i in range(n):
		text_i = (data.get("text", [""] * n)[i] or "").strip()
		conf_raw = data.get("conf", ["-1"] * n)[i]
		try:
			conf = float(conf_raw)
		except Exception:
			conf = -1.0

		if text_i != "" and conf >= conf_threshold:
			x = int(data.get("left", [0] * n)[i])
			y = int(data.get("top", [0] * n)[i])
			w = int(data.get("width", [0] * n)[i])
			h = int(data.get("height", [0] * n)[i])

			# Gambar kotak hijau di sekitar teks
			cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
			# Teks kecil di atas kotak (opsional)
			cv2.putText(
				annotated,
				text_i[:25],
				(x, max(0, y - 5)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(0, 0, 255),
				1,
				cv2.LINE_AA,
			)
			count += 1

	return annotated, count


def save_text(text: str, path: str = "hasil_teks.txt") -> None:
	"""Simpan teks hasil OCR ke file UTF-8."""
	with open(path, "w", encoding="utf-8") as f:
		f.write(text)


def make_fallback_annotated(img_bgr: np.ndarray, message: str) -> np.ndarray:
	"""
	Buat gambar anotasi sederhana yang menjelaskan mode fallback (tanpa Tesseract).
	"""
	annotated = img_bgr.copy()
	h, w = annotated.shape[:2]
	banner_h = max(30, h // 18)
	# Banner merah semi-transparan di atas
	overlay = annotated.copy()
	cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 255), thickness=-1)
	alpha = 0.5
	annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
	# Teks putih di banner
	cv2.putText(
		annotated,
		message,
		(10, int(banner_h * 0.7)),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return annotated


def main() -> None:
	print("=== Identifikasi Teks pada Gambar (OCR) ===")
	print("Masukkan nama file gambar (contoh: nota.jpg atau teks.png)")
	img_path = input("Nama file gambar: ").strip().strip('"').strip("'")

	if not os.path.isfile(img_path):
		print(f"[!] File tidak ditemukan: {img_path}")
		print("Pastikan path benar dan file berada di folder kerja ini atau berikan path lengkap.")
		sys.exit(1)

	# Baca gambar
	img = cv2.imread(img_path)
	if img is None:
		print("[!] Gagal membaca gambar. Format mungkin tidak didukung atau file korup.")
		sys.exit(1)

	# Pastikan Tesseract terpasang dan path terdeteksi (khusus Windows)
	detected = ensure_tesseract_cmd()
	if os.name == "nt" and detected is None:
		print("[!] Tidak menemukan tesseract.exe di lokasi umum.")
		print("    Silakan install Tesseract OCR dan/atau set path manual di kode: pytesseract.pytesseract.tesseract_cmd")

	# Preprocessing
	processed, gray = preprocess_image(img)

	# OCR
	ocr_ok = True
	try:
		text = perform_ocr(processed, lang="eng")  # ganti ke 'ind' jika paket bahasa Indonesia terpasang
	except pytesseract.TesseractNotFoundError:
		ocr_ok = False
		text = ""
		print("[!] Tesseract OCR tidak ditemukan. Melewati langkah OCR dan bounding box (mode fallback).")
	except Exception as e:
		ocr_ok = False
		text = ""
		print(f"[!] Terjadi kesalahan saat OCR: {e}. Melewati langkah OCR (mode fallback).")

	# Tampilkan hasil teks di terminal
	print("\n=== Hasil OCR ===")
	if text:
		print(text)
	else:
		print("[Kosong] Tidak ada teks terdeteksi.")

	# Simpan hasil teks
	try:
		save_text(text, "hasil_teks.txt")
		print("\n[+] Teks tersimpan ke: hasil_teks.txt")
	except Exception as e:
		print(f"[!] Gagal menyimpan teks: {e}")

	# Gambar bounding box dan simpan gambar hasil
	try:
		if ocr_ok:
			annotated, n_boxes = draw_bounding_boxes(img, processed, lang="eng", conf_threshold=60)
			out_img_path = "hasil_deteksi.jpg"
			cv2.imwrite(out_img_path, annotated)
			print(f"[+] Gambar hasil deteksi tersimpan ke: {out_img_path} (kotak: {n_boxes})")
		else:
			# Mode fallback: tidak ada bounding box, beri banner info
			annotated = make_fallback_annotated(
				img, "Fallback: Tesseract tidak tersedia - OCR dilewati"
			)
			out_img_path = "hasil_deteksi.jpg"
			cv2.imwrite(out_img_path, annotated)
			print(f"[+] (Fallback) Gambar tersimpan ke: {out_img_path}")
	except Exception as e:
		print(f"[!] Gagal membuat/menyimpan gambar hasil deteksi: {e}")
		annotated = None

	# Tampilkan gambar (opsional, dapat gagal pada lingkungan headless)
	try:
		cv2.imshow("Gambar Asli", img)
		cv2.imshow("Hasil Preprocessing", processed)
		if annotated is not None:
			cv2.imshow("Deteksi Teks (Bounding Box)", annotated)
		print("\nTutup jendela gambar dengan menekan tombol apa saja...")
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	except cv2.error:
		print("[!] Tidak dapat menampilkan jendela gambar (mungkin lingkungan tanpa GUI). Gambar sudah disimpan.")


if __name__ == "__main__":
	main()

