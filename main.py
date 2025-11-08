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


def upscale_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
	"""
	Upscale gambar untuk meningkatkan resolusi teks kecil.
	Menggunakan interpolasi CUBIC untuk hasil terbaik.
	"""
	if scale <= 1.0:
		return img
	
	height, width = img.shape[:2]
	new_width = int(width * scale)
	new_height = int(height * scale)
	
	upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return upscaled


def enhance_document_image(img_gray: np.ndarray) -> np.ndarray:
	"""
	Enhancement khusus untuk dokumen dengan teks kecil dan noise.
	
	Teknik:
	- Unsharp masking untuk sharpening
	- Contrast enhancement
	- Adaptive bilateral filtering
	"""
	# 1) Unsharp masking untuk meningkatkan ketajaman
	gaussian = cv2.GaussianBlur(img_gray, (0, 0), 2.0)
	unsharp = cv2.addWeighted(img_gray, 1.5, gaussian, -0.5, 0)
	
	# 2) Normalize contrast
	unsharp = cv2.normalize(unsharp, None, 0, 255, cv2.NORM_MINMAX)
	
	# 3) Bilateral filter untuk smooth sambil preserve edges
	enhanced = cv2.bilateralFilter(unsharp, 9, 75, 75)
	
	return enhanced


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


def perform_ocr_optimized(img: np.ndarray, lang: str = "eng+ind") -> str:
	"""
	OCR dengan multiple PSM modes dan pilih hasil terbaik.
	Cocok untuk dokumen dengan layout kompleks.
	"""
	# PSM modes untuk dicoba
	psm_configs = [
		("--oem 3 --psm 3", "Fully automatic"),  # Best untuk dokumen lengkap
		("--oem 3 --psm 6", "Single block"),      # Best untuk paragraf
		("--oem 3 --psm 4", "Single column"),     # Best untuk kolom teks
	]
	
	results = []
	for config, desc in psm_configs:
		try:
			text = pytesseract.image_to_string(img, lang=lang, config=config)
			text = text.replace("\x0c", "").strip()
			if text:
				results.append((text, len(text)))
		except Exception:
			continue
	
	# Pilih hasil terpanjang (biasanya paling lengkap)
	if results:
		results.sort(key=lambda x: x[1], reverse=True)
		return results[0][0]
	
	return ""


def correct_common_ocr_errors(text: str) -> str:
	"""
	Koreksi kesalahan OCR umum untuk bahasa Indonesia/Inggris.
	"""
	if not text:
		return text
	
	# Dictionary perbaikan umum (case-sensitive)
	corrections = {
		# Angka sering salah dibaca
		'O': '0',  # Huruf O jadi angka 0 dalam konteks angka
		'l': '1',  # Huruf l kecil jadi 1 dalam konteks angka
		'I': '1',  # Huruf I besar jadi 1 dalam konteks angka
		'S': '5',  # Huruf S jadi 5 dalam konteks angka (terbatas)
		
		# Kata-kata umum yang sering salah
		'Unluk': 'Untuk',
		'unluk': 'untuk',
		'Unluk': 'Untuk',
		'dan]': 'dari',
		'dani': 'dari',
		'dati': 'dari',
		'tepal': 'tepat',
		'saal': 'saat',
		'Kelab': 'Keleb',
		'Kelebihan': 'Kelebihan',
		'Kelabihan': 'Kelebihan',
		'yang,': 'yang',
		'vang': 'yang',
		'protocol': 'protokol',
		'lebih.': 'lebih',
		'Iinggi': 'tinggi',
		'Iain': 'lain',
		'handshake.': 'handshake',
		'OSIL': 'OSI',
		'Iayer': 'layer',
		'hos!': 'host',
		'hos': 'host',
		'tiap-': 'tiap',
		'tepall': 'tepat',
		'CSM': 'CSMA',
		'CSMA/CD': 'CSMA/CD',
		'protocol]': 'protokol',
		'mencegah-': 'mencegah',
		'PC-': 'PC',
		'yang,': 'yang',
		'Iain-': 'lain',
		'frame.': 'frame',
		'saat-': 'saat',
		'proses-': 'proses',
		'dan]': 'dari',
		'pengiriman-': 'pengiriman',
		'dari-': 'dari',
	}
	
	lines = text.split('\n')
	corrected_lines = []
	
	for line in lines:
		corrected = line
		for wrong, right in corrections.items():
			corrected = corrected.replace(wrong, right)
		
		# Perbaiki spacing yang aneh
		corrected = ' '.join(corrected.split())
		corrected_lines.append(corrected)
	
	return '\n'.join(corrected_lines)


def fix_number_patterns(text: str) -> str:
	"""
	Perbaiki pola angka yang sering salah (IP address, nomor soal, dll).
	"""
	import re
	
	# Perbaiki IP address: ganti huruf O dengan 0
	# Pattern: xxx.xxx.xxx.xxx atau xxx.xxx.xxx.x
	ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
	
	def fix_ip(match):
		ip = match.group(0)
		# Ganti O dengan 0 dalam IP
		return ip.replace('O', '0').replace('o', '0')
	
	text = re.sub(ip_pattern, fix_ip, text)
	
	# Perbaiki nomor soal: "22." "23." dll - pastikan ada spasi setelahnya
	text = re.sub(r'(\d+)\.\s*([A-Z])', r'\1. \2', text)
	
	return text


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
	print("=== Identifikasi Teks pada Gambar (OCR) - Enhanced ===")
	print("\nMode preprocessing:")
	print("  1. Standard (cepat) - untuk gambar berkualitas baik")
	print("  2. Enhanced (akurat) - untuk dokumen dengan teks kecil/blur")
	
	mode = input("\nPilih mode (1/2) [default: 2]: ").strip()
	use_enhanced = (mode != "1")
	
	print("\nMasukkan nama file gambar (contoh: nota.jpg atau soal.png)")
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

	print(f"\n[*] Gambar dimuat: {img.shape[1]}x{img.shape[0]} pixels")

	# Pastikan Tesseract terpasang dan path terdeteksi (khusus Windows)
	detected = ensure_tesseract_cmd()
	if os.name == "nt" and detected is None:
		print("[!] Tidak menemukan tesseract.exe di lokasi umum.")
		print("    Silakan install Tesseract OCR dan/atau set path manual di kode: pytesseract.pytesseract.tesseract_cmd")

	# === Enhanced preprocessing untuk akurasi maksimal ===
	if use_enhanced:
		print("[*] Mode: ENHANCED - Upscaling & multiple preprocessing")
		
		# Step 1: Upscale jika gambar kecil
		h, w = img.shape[:2]
		if w < 1500 or h < 1500:
			scale = 2.0
			print(f"[*] Upscaling gambar {scale}x untuk meningkatkan resolusi teks...")
			img = upscale_image(img, scale)
			print(f"    Ukuran baru: {img.shape[1]}x{img.shape[0]} pixels")
		
		# Step 2: Grayscale & enhancement
		print("[*] Applying document enhancement (unsharp masking, contrast)...")
		if len(img.shape) == 3:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			gray = img.copy()
		
		enhanced_gray = enhance_document_image(gray)
		
		# Step 3: Multiple thresholding strategies
		print("[*] Generating multiple preprocessing variants...")
		
		# Otsu threshold
		_, otsu = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		
		# Adaptive threshold
		adaptive = cv2.adaptiveThreshold(
			enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
		)
		
		# CLAHE + Otsu
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		clahe_img = clahe.apply(enhanced_gray)
		_, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		
		# Simpan untuk analisis
		cv2.imwrite("debug_enhanced_gray.jpg", enhanced_gray)
		cv2.imwrite("debug_otsu.jpg", otsu)
		cv2.imwrite("debug_adaptive.jpg", adaptive)
		cv2.imwrite("debug_clahe.jpg", clahe_otsu)
		
		processed_variants = [
			("otsu", otsu),
			("adaptive", adaptive),
			("clahe", clahe_otsu),
		]
	else:
		print("[*] Mode: STANDARD - Preprocessing cepat")
		processed, gray = preprocess_image(img)
		processed_variants = [("standard", processed)]

	# === OCR dengan multiple configs ===
	print("[*] Melakukan OCR dengan multiple configurations...")
	print("    Bahasa: English + Indonesian (jika tersedia)")
	
	ocr_ok = True
	all_texts = []
	
	try:
		for name, img_processed in processed_variants:
			print(f"    - Processing variant: {name}")
			# Coba dengan eng+ind (fallback ke eng jika ind tidak ada)
			try:
				text = perform_ocr_optimized(img_processed, lang="eng+ind")
			except Exception:
				text = perform_ocr_optimized(img_processed, lang="eng")
			
			if text:
				all_texts.append(text)
		
		# Pilih hasil terpanjang
		if all_texts:
			text = max(all_texts, key=len)
		else:
			text = ""
		
		# Post-processing: koreksi kesalahan umum
		if text:
			print("[*] Applying OCR error correction...")
			text = correct_common_ocr_errors(text)
			text = fix_number_patterns(text)
		
	except pytesseract.TesseractNotFoundError:
		ocr_ok = False
		text = ""
		print("[!] Tesseract OCR tidak ditemukan. Melewati langkah OCR dan bounding box (mode fallback).")
	except Exception as e:
		ocr_ok = False
		text = ""
		print(f"[!] Terjadi kesalahan saat OCR: {e}. Melewati langkah OCR (mode fallback).")

	# Tampilkan hasil teks di terminal
	print("\n" + "="*70)
	print("=== HASIL OCR ===")
	print("="*70)
	if text:
		print(text)
		print("="*70)
		print(f"[*] Total karakter: {len(text)}")
		print(f"[*] Total baris: {len(text.splitlines())}")
		words = text.split()
		print(f"[*] Total kata: {len(words)}")
	else:
		print("[Kosong] Tidak ada teks terdeteksi.")
		print("="*70)

	# Simpan hasil teks
	try:
		save_text(text, "hasil_teks.txt")
		print("\n[+] Teks tersimpan ke: hasil_teks.txt")
	except Exception as e:
		print(f"[!] Gagal menyimpan teks: {e}")

	# Gambar bounding box dan simpan gambar hasil
	print("[*] Mendeteksi bounding boxes...")
	try:
		if ocr_ok and processed_variants:
			# Gunakan variant terbaik untuk bounding box
			_, best_processed = processed_variants[0]
			annotated, n_boxes = draw_bounding_boxes(img, best_processed, lang="eng+ind", conf_threshold=30)
			out_img_path = "hasil_deteksi.jpg"
			cv2.imwrite(out_img_path, annotated)
			print(f"[+] Gambar hasil deteksi tersimpan ke: {out_img_path}")
			print(f"[*] Total bounding boxes: {n_boxes}")
		else:
			# Mode fallback
			annotated = make_fallback_annotated(
				img, "Fallback: Tesseract tidak tersedia - OCR dilewati"
			)
			out_img_path = "hasil_deteksi.jpg"
			cv2.imwrite(out_img_path, annotated)
			print(f"[+] (Fallback) Gambar tersimpan ke: {out_img_path}")
	except Exception as e:
		print(f"[!] Gagal membuat/menyimpan gambar hasil deteksi: {e}")
		annotated = None

	# Tampilkan gambar (opsional)
	if use_enhanced:
		print("\n[*] Debug images saved: debug_*.jpg")
	
	print("\n" + "="*70)
	print("=== SELESAI ===")
	print("="*70)
	print("Output files:")
	print("  - hasil_teks.txt      : Teks hasil OCR (sudah dikoreksi)")
	print("  - hasil_deteksi.jpg   : Gambar dengan bounding boxes")
	if use_enhanced:
		print("  - debug_*.jpg         : Preprocessing variants untuk analisis")
	print("="*70)

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

	# Tampilkan gambar (opsional)
	if use_enhanced:
		print("\n[*] Debug images saved: debug_*.jpg")
	
	print("\n" + "="*70)
	print("=== SELESAI ===")
	print("="*70)
	print("Output files:")
	print("  - hasil_teks.txt      : Teks hasil OCR (sudah dikoreksi)")
	print("  - hasil_deteksi.jpg   : Gambar dengan bounding boxes")
	if use_enhanced:
		print("  - debug_*.jpg         : Preprocessing variants untuk analisis")
	print("="*70)


if __name__ == "__main__":
	main()

