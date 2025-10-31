# 📊 NLP - One for All

Aplikasi web interaktif untuk berbagai tugas Natural Language Processing (NLP) menggunakan Streamlit.

## 🚀 Fitur

1. **🔍 Feature Extraction (TF-IDF)**
   - Ekstraksi fitur menggunakan metode TF-IDF
   - Menghasilkan matriks TF-IDF dan vocabulary
   - Visualisasi: Heatmap, Word Cloud, Bar Chart
   - Upload file CSV/TXT atau input manual
   - Export hasil ke CSV/Excel

2. **📑 Inverted Index**
   - Membuat struktur data inverted index sederhana
   - Memetakan kata ke dokumen yang mengandung kata tersebut
   - Fitur pencarian: Pencarian kata tunggal dan multi-kata (AND/OR)
   - Highlighting kata yang ditemukan
   - Statistik lengkap dan visualisasi
   - Upload file CSV/TXT atau input manual
   - Export hasil ke CSV/Excel

3. **🎯 Model LDA**
   - Pelatihan model Latent Dirichlet Allocation
   - Mendukung Bahasa Indonesia dan Inggris
   - Visualisasi interaktif pyLDAvis
   - Word Cloud per topik
   - Distribusi topik per dokumen
   - Upload file CSV/TXT atau input manual
   - Export hasil ke CSV/Excel

4. **🤖 Model BERTopic**
   - Analisis topik menggunakan BERTopic dengan embedding transformer
   - Mendukung Bahasa Indonesia dan Inggris
   - Visualisasi interaktif (jarak antar topik, bar chart)
   - Auto-adjustment untuk dataset kecil
   - Upload file CSV/TXT atau input manual
   - Export hasil ke CSV/Excel

## 📦 Instalasi

1. Clone atau download repository ini

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser default Anda pada `http://localhost:8501`

## 💻 Requirements

- Python 3.8 atau lebih tinggi
- Streamlit
- Pandas
- Scikit-learn
- NLTK
- Gensim
- BERTopic
- Plotly
- Sentence-transformers

## 📝 Catatan

- Untuk pertama kali menjalankan BERTopic, model embedding akan diunduh secara otomatis (membutuhkan waktu beberapa menit)
- Pastikan koneksi internet tersedia untuk mengunduh model BERTopic
- NLTK akan otomatis mengunduh data yang diperlukan (punkt, stopwords)

## 🎯 Penggunaan

1. Pilih program dari menu sidebar
2. Masukkan dokumen sesuai format yang diminta
3. Klik tombol untuk memproses
4. Lihat hasilnya!

## 📄 Lisensi

Aplikasi ini dibuat untuk tujuan pembelajaran dan penggunaan umum.

