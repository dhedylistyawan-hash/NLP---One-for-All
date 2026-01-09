# 📚 Dokumentasi Fitur Baru - Format Standar

## Menu 5: Word Vector Representations

•	**Fungsi**: Menu ini melatih model Word Vector (Word2Vec) untuk menghasilkan representasi vektor numerik dari kata-kata dalam kumpulan dokumen. Setiap kata direpresentasikan sebagai vektor dalam ruang dimensi tinggi, di mana kata-kata yang mirip secara semantik akan memiliki vektor yang dekat. Model ini memungkinkan pencarian kata yang mirip dan perhitungan similarity (kemiripan) antara dua kata berdasarkan konteks penggunaannya dalam dokumen.

•	**Algoritma**: Prosesnya meliputi preprocessing (tokenisasi, lowercasing, filtering kata dengan panjang > 2 karakter dan hanya alfabet), pelatihan model Word2Vec menggunakan arsitektur CBOW (Continuous Bag of Words), dan ekstraksi vektor embedding. CBOW memprediksi kata target berdasarkan konteks (kata-kata di sekitarnya) dengan menggunakan neural network satu hidden layer. Similarity dihitung menggunakan cosine similarity antara vektor kata.

•	**Implementasi Kode**: Teks di-preprocess menggunakan `word_tokenize` dari NLTK untuk tokenisasi, kemudian difilter untuk hanya mengambil kata alfabet dengan panjang > 2. Model menggunakan `Word2Vec` dari library Gensim dengan parameter `vector_size` (50-300), `window=5` (konteks 5 kata), `min_count=1`, `workers=4`, dan `sg=0` (CBOW). Pencarian kata mirip menggunakan `model.wv.most_similar()` dan perhitungan similarity menggunakan `model.wv.similarity()`.

•	**Visualisasi**:
  - Bar chart horizontal menampilkan top 10 kata yang mirip dengan kata input, dengan similarity score sebagai sumbu X.
  - Metrics display menampilkan ukuran vektor, jumlah kata dalam vocabulary, dan jumlah dokumen yang diproses.
  - Tabel vocabulary (top 50 kata) dalam expander.

•	**Fitur Menu**: Upload file (CSV/TXT) atau input manual, pilihan model (Word2Vec/FastText), slider untuk menentukan ukuran vektor (50-300), pencarian kata yang mirip, perhitungan similarity antara dua kata, tampilan vocabulary, dan ekspor hasil ke CSV/Excel.

•	**Contoh Hasil**: 
  - Input: "Saya suka belajar machine learning. Deep learning adalah cabang dari machine learning."
  - Output Informasi Model: Ukuran Vektor: 100, Jumlah Kata: 15, Jumlah Dokumen: 2
  - Cari kata mirip dengan "machine": learning (0.8234), deep (0.7123), neural (0.6891)
  - Similarity antara "machine" dan "learning": 0.8234 (Kedua kata sangat mirip!)

---

## Menu 6: Recurrent Neural Network (RNN)

•	**Fungsi**: Menu ini melatih model RNN menggunakan LSTM (Long Short-Term Memory) untuk dua task utama: Text Classification (mengklasifikasikan teks ke dalam kategori tertentu seperti positif/negatif) dan Text Generation (menghasilkan teks baru berdasarkan seed text). LSTM mampu memproses urutan teks sambil mempertahankan informasi dari langkah sebelumnya, sehingga cocok untuk memahami konteks dalam teks.

•	**Algoritma**: Untuk Text Classification, prosesnya meliputi preprocessing (tokenisasi menggunakan Keras Tokenizer, padding sequence untuk menyamakan panjang), pembuatan model Sequential dengan Embedding layer, dua LSTM layers (64 dan 32 units), dan Dense layer dengan softmax activation. Model dilatih menggunakan optimizer Adam dengan loss function sparse_categorical_crossentropy. Untuk Text Generation, menggunakan model pre-trained GPT-2 dari Transformers library yang menghasilkan teks berdasarkan seed text.

•	**Implementasi Kode**: Teks di-tokenize menggunakan `Tokenizer()` dari Keras, kemudian diubah menjadi sequences dan di-pad menggunakan `pad_sequences()`. Model dibangun menggunakan `Sequential()` dengan `Embedding()`, `LSTM()` layers, dan `Dense()` output layer. Pelatihan menggunakan `model.fit()` dengan 10 epochs, batch size 32, dan train/test split 80/20. Untuk text generation, menggunakan `pipeline("text-generation", model="gpt2")` dari Transformers.

•	**Visualisasi**:
  - Line chart training history menampilkan accuracy training vs validation per epoch (sumbu X: epoch, sumbu Y: accuracy 0-1).
  - Line chart training history menampilkan loss training vs validation per epoch (sumbu X: epoch, sumbu Y: loss).
  - Metrics display menampilkan training accuracy dan validation accuracy akhir.
  - Tabel probabilitas semua kategori untuk prediksi teks baru.

•	**Fitur Menu**: 
  - **Text Classification**: Upload file CSV dengan kolom 'text' dan 'label' atau input manual dengan format `teks|label`, pelatihan model LSTM, visualisasi training history, prediksi teks baru dengan confidence score, dan tampilan probabilitas semua kategori.
  - **Text Generation**: Input seed text, slider untuk panjang teks yang di-generate (10-100), generate teks menggunakan GPT-2, dan highlight seed text dalam hasil.

•	**Contoh Hasil**: 
  - **Text Classification Input**: "Saya suka belajar machine learning|positif", "Saya tidak suka matematika|negatif"
  - **Output**: Training Accuracy: 0.9500, Validation Accuracy: 0.8000
  - **Prediksi**: Input "Saya suka belajar AI" → Prediksi: **positif** (Confidence: 0.9234)
  - **Text Generation**: Seed "Machine learning adalah" → Generate: "Machine learning adalah cabang dari artificial intelligence yang memungkinkan sistem untuk belajar dari data..."

---

## Menu 7: Sequence to Sequence (Seq2Seq)

•	**Fungsi**: Menu ini menggunakan model Seq2Seq berbasis Transformer untuk tiga task: Translation (menerjemahkan teks dari satu bahasa ke bahasa lain), Summarization (membuat ringkasan dari teks panjang), dan Question Answering (menjawab pertanyaan berdasarkan konteks yang diberikan). Model Seq2Seq menggunakan arsitektur Encoder-Decoder dengan attention mechanism untuk menangkap hubungan antara input dan output sequence.

•	**Algoritma**: Prosesnya menggunakan model pre-trained Transformer dengan arsitektur Encoder-Decoder. Encoder memproses input sequence menggunakan multi-head self-attention dan feed-forward networks. Decoder menggunakan masked self-attention dan encoder-decoder attention untuk menghasilkan output sequence. Untuk Translation menggunakan model Helsinki-NLP/opus-mt (MarianMT), untuk Summarization menggunakan facebook/bart-large-cnn (BART), dan untuk Question Answering menggunakan distilbert-base-uncased (DistilBERT).

•	**Implementasi Kode**: Menggunakan `pipeline()` dari Transformers library dengan task-specific models. Translation menggunakan `pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")`, Summarization menggunakan `pipeline("summarization", model="facebook/bart-large-cnn")` dengan parameter `max_length` dan `min_length`, dan Question Answering menggunakan `pipeline("question-answering")` dengan input `question` dan `context`. Semua pipeline menggunakan `device=-1` untuk CPU.

•	**Visualisasi**:
  - **Translation**: Side-by-side comparison menampilkan teks sumber dan hasil terjemahan dalam dua kolom.
  - **Summarization**: Metrics comparison menampilkan panjang teks asli (kata) dan panjang ringkasan (kata) dalam dua kolom.
  - **Question Answering**: Highlight jawaban dalam konteks dengan background kuning, dan menampilkan confidence score.

•	**Fitur Menu**: 
  - **Translation**: Pilih bahasa sumber dan target (English, Indonesian, French, German, Spanish), input teks untuk diterjemahkan, dan tampilan hasil terjemahan side-by-side.
  - **Summarization**: Input teks panjang, slider untuk panjang maksimal (30-150) dan minimal (10-50) ringkasan, dan tampilan ringkasan dengan perbandingan panjang.
  - **Question Answering**: Input konteks (teks referensi) dan pertanyaan, tampilan jawaban dengan confidence score, dan highlight jawaban dalam konteks.

•	**Contoh Hasil**: 
  - **Translation**: Input "Hello, how are you?" (English → Indonesian) → Output "Halo, apa kabar?"
  - **Summarization**: Input teks 85 kata → Output ringkasan 18 kata (rasio kompresi 21.2%)
  - **Question Answering**: Konteks "Machine learning is a method..." + Pertanyaan "What is machine learning?" → Jawaban: "a method of data analysis that automates analytical model building" (Confidence: 0.9234)

---

## Menu 8: Transformers

•	**Fungsi**: Menu ini menggunakan model Transformer pre-trained untuk berbagai task NLP: Sentiment Analysis (menganalisis sentimen teks menjadi positif/negatif/neutral), Named Entity Recognition atau NER (mengidentifikasi dan mengekstrak entitas seperti nama orang, organisasi, lokasi), Text Classification (mengklasifikasikan teks ke kategori tertentu), dan Zero-Shot Classification (klasifikasi tanpa training, hanya dengan memberikan label). Model Transformer menggunakan self-attention mechanism untuk memahami konteks dalam teks.

•	**Algoritma**: Prosesnya menggunakan model pre-trained Transformer dengan arsitektur self-attention. Attention mechanism menghitung hubungan antara semua posisi dalam sequence menggunakan query, key, dan value vectors. Multi-head attention memungkinkan model menangkap berbagai aspek hubungan. Untuk Sentiment Analysis menggunakan distilbert-base-uncased-finetuned-sst-2-english, untuk NER menggunakan dbmdz/bert-large-cased-finetuned-conll03-english, dan untuk Classification menggunakan facebook/bart-large-mnli (zero-shot).

•	**Implementasi Kode**: Menggunakan `pipeline()` dari Transformers library untuk setiap task. Sentiment Analysis menggunakan `pipeline("sentiment-analysis")`, NER menggunakan `pipeline("ner", aggregation_strategy="simple")`, dan Zero-Shot Classification menggunakan `pipeline("zero-shot-classification")` dengan input teks dan `candidate_labels`. Semua pipeline menggunakan `device=-1` untuk CPU. Hasil diproses dan ditampilkan dalam DataFrame untuk visualisasi dan export.

•	**Visualisasi**:
  - **Sentiment Analysis**: Bar chart vertikal distribusi sentiment dengan warna hijau (positive), merah (negative), biru (neutral), sumbu X: sentiment label, sumbu Y: jumlah dokumen.
  - **NER**: Horizontal bar chart distribusi entity types (PER, ORG, LOC, MISC) dengan warna coral, sumbu X: jumlah, sumbu Y: entity type.
  - **Zero-Shot Classification**: Horizontal bar chart scores untuk semua label dengan warna steel blue, sumbu X: score (0-1), sumbu Y: label.

•	**Fitur Menu**: Upload file (CSV/TXT) atau input manual, pilihan task (Sentiment Analysis, NER, Text Classification, Zero-Shot Classification), untuk Text Classification dan Zero-Shot: input kategori custom (dipisahkan koma), visualisasi distribusi hasil, dan ekspor hasil ke CSV/Excel.

•	**Contoh Hasil**: 
  - **Sentiment Analysis**: Input "I love this product!" → Output: Label: POSITIVE, Score: 0.9998
  - **NER**: Input "Barack Obama was born in Hawaii" → Output: Entity "Barack Obama" (PER, 0.9987), "Hawaii" (LOC, 0.9876)
  - **Zero-Shot Classification**: Input "This is a great movie" dengan kategori "positive, negative, neutral" → Output: Prediksi: positive (Score: 0.9234), dengan tabel probabilitas semua kategori.

---

## Ringkasan Perbandingan

| Menu | Task Utama | Model/Algorithm | Library | Input Format |
|------|-----------|-----------------|---------|--------------|
| **Word Vector** | Word Embedding | Word2Vec (CBOW) | Gensim | Dokumen teks |
| **RNN** | Classification & Generation | LSTM, GPT-2 | TensorFlow, Transformers | Teks + Label (classification) atau Seed text (generation) |
| **Seq2Seq** | Translation, Summarization, QA | Transformer (MarianMT, BART, DistilBERT) | Transformers | Teks (sesuai task) |
| **Transformers** | Sentiment, NER, Classification | Pre-trained Transformer | Transformers | Teks (sesuai task) |

---

**Catatan**: Semua model pre-trained (Seq2Seq dan Transformers) akan diunduh otomatis saat pertama kali digunakan dan memerlukan koneksi internet. Proses download membutuhkan waktu beberapa menit tergantung kecepatan internet.

