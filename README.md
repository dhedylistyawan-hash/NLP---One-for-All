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

5. **🔤 Word Vector Representations**
   - Melatih model Word2Vec dan FastText
   - Mencari kata yang mirip (word similarity)
   - Menghitung similarity antara dua kata
   - Visualisasi similarity scores
   - Upload file CSV/TXT atau input manual
   - Export vocabulary ke CSV/Excel

6. **🧠 Recurrent Neural Network (RNN)**
   - Text Classification menggunakan LSTM
   - Text Generation menggunakan GPT-2
   - Visualisasi training history (accuracy & loss)
   - Prediksi teks baru
   - Upload file CSV dengan format text|label

7. **🔄 Sequence to Sequence (Seq2Seq)**
   - Translation (Terjemahan) antar bahasa
   - Summarization (Ringkasan teks panjang)
   - Question Answering (Menjawab pertanyaan dari konteks)
   - Menggunakan model transformer pre-trained

8. **⚡ Transformers**
   - Sentiment Analysis (Analisis sentimen)
   - Named Entity Recognition (NER)
   - Text Classification
   - Zero-Shot Classification
   - Menggunakan model transformer modern dari Hugging Face

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
- Gensim (Word2Vec, FastText)
- BERTopic
- Plotly
- Sentence-transformers
- Transformers (Hugging Face)
- TensorFlow/Keras (untuk RNN)
- PyTorch (untuk Transformers)

## 📝 Catatan

- Untuk pertama kali menjalankan BERTopic, model embedding akan diunduh secara otomatis (membutuhkan waktu beberapa menit)
- Model Transformers (Hugging Face) akan diunduh otomatis saat pertama kali digunakan
- Pastikan koneksi internet tersedia untuk mengunduh model-model pre-trained
- NLTK akan otomatis mengunduh data yang diperlukan (punkt, stopwords)
- Untuk RNN dan Transformers, pastikan RAM cukup (minimal 4GB direkomendasikan)

## 🎯 Penggunaan

1. Pilih program dari menu sidebar
2. Masukkan dokumen sesuai format yang diminta
3. Klik tombol untuk memproses
4. Lihat hasilnya!

## 📄 Lisensi

Aplikasi ini dibuat untuk tujuan pembelajaran dan penggunaan umum.

