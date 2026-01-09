# 📚 Dokumentasi Lengkap Fitur Baru NLP - One for All

## 🔤 1. WORD VECTOR REPRESENTATIONS

### 📋 Fungsi Menu
Menu ini melatih model Word Vector (Word2Vec) untuk menghasilkan representasi vektor numerik dari kata-kata. Setiap kata direpresentasikan sebagai vektor dalam ruang dimensi tinggi, di mana kata-kata yang mirip secara semantik akan memiliki vektor yang dekat.

**Kegunaan:**
- Mencari kata yang mirip secara semantik
- Menghitung similarity (kemiripan) antara dua kata
- Memahami hubungan semantik antar kata dalam korpus
- Dapat digunakan untuk downstream tasks seperti text classification, clustering, dll

### 🧮 Algoritma

#### Word2Vec (CBOW - Continuous Bag of Words)
1. **Input**: Kumpulan dokumen teks
2. **Preprocessing**: 
   - Tokenisasi (memecah teks menjadi kata-kata)
   - Lowercasing (mengubah ke huruf kecil)
   - Filtering (hanya kata dengan panjang > 2 karakter dan alfabet)
3. **Training**:
   - **CBOW**: Memprediksi kata target berdasarkan konteks (kata-kata di sekitarnya)
   - **Skip-gram**: Memprediksi konteks berdasarkan kata target
   - Menggunakan neural network dengan satu hidden layer
   - Output: Vektor embedding untuk setiap kata
4. **Parameter**:
   - `vector_size`: Dimensi vektor (50-300)
   - `window`: Jumlah kata di kiri-kanan (default: 5)
   - `min_count`: Frekuensi minimum kata (default: 1)
   - `sg`: 0 untuk CBOW, 1 untuk Skip-gram

**Rumus Similarity (Cosine Similarity):**
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

### 💻 Implementasi Kode

```python
# Preprocessing
def preprocess_for_wv(texts):
    processed = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
        if tokens:
            processed.append(tokens)
    return processed

# Training Model
model = Word2Vec(
    sentences=processed_texts,
    vector_size=vector_size,  # 50-300
    window=5,
    min_count=1,
    workers=4,
    sg=0  # CBOW
)

# Mencari kata mirip
similar_words = model.wv.most_similar(word, topn=10)

# Menghitung similarity
similarity = model.wv.similarity(word1, word2)
```

**Library yang digunakan:**
- `gensim.models.Word2Vec`: Untuk training model Word2Vec
- `nltk.word_tokenize`: Untuk tokenisasi teks

### 📊 Visualisasi

1. **Bar Chart Similarity Scores**
   - Menampilkan top 10 kata yang mirip dengan kata input
   - Horizontal bar chart dengan similarity score
   - Warna: Steel blue
   - Format: `matplotlib.pyplot`

2. **Informasi Model (Metrics)**
   - Ukuran vektor
   - Jumlah kata dalam vocabulary
   - Jumlah dokumen yang diproses

### ⚙️ Fitur Menu

1. **Pemilihan Model**
   - Word2Vec (tersedia)
   - FastText (opsional, jika terinstall)

2. **Parameter yang Dapat Disesuaikan**
   - Ukuran vektor: 50-300 (slider)
   - Input: Manual atau Upload File (CSV/TXT)

3. **Fitur Utama**
   - ✅ Cari kata yang mirip (most similar words)
   - ✅ Hitung similarity antara dua kata
   - ✅ Tampilkan vocabulary (top 50 kata)
   - ✅ Export vocabulary ke CSV/Excel

4. **Input/Output**
   - Input: Dokumen teks (minimal 3 dokumen)
   - Output: Model Word Vector, similarity scores, vocabulary list

### 📝 Contoh Hasilnya

**Input:**
```
Saya suka belajar machine learning.
Deep learning adalah cabang dari machine learning.
Natural language processing menggunakan neural network.
```

**Output:**

**Informasi Model:**
- Ukuran Vektor: 100
- Jumlah Kata: 25
- Jumlah Dokumen: 3

**Cari Kata Mirip dengan "machine":**
| Kata | Similarity Score |
|------|------------------|
| learning | 0.8234 |
| neural | 0.7123 |
| network | 0.6891 |
| deep | 0.6543 |
| processing | 0.6123 |

**Similarity antara "machine" dan "learning":**
- Similarity Score: **0.8234**
- Interpretasi: ✅ Kedua kata sangat mirip!

---

## 🧠 2. RECURRENT NEURAL NETWORK (RNN)

### 📋 Fungsi Menu
Menu ini melatih model RNN menggunakan LSTM (Long Short-Term Memory) untuk dua task utama:
1. **Text Classification**: Mengklasifikasikan teks ke dalam kategori tertentu
2. **Text Generation**: Menghasilkan teks baru berdasarkan seed text

**Kegunaan:**
- Klasifikasi sentimen, kategori dokumen, dll
- Generasi teks otomatis
- Memahami urutan dan konteks dalam teks

### 🧮 Algoritma

#### LSTM (Long Short-Term Memory)
1. **Architecture**:
   ```
   Input → Embedding Layer → LSTM Layer 1 (64 units, return_sequences=True) 
   → LSTM Layer 2 (32 units) → Dense Layer (softmax) → Output
   ```

2. **Proses Training**:
   - **Tokenization**: Mengubah teks menjadi sequence of integers
   - **Padding**: Menyamakan panjang sequence
   - **Embedding**: Mengubah integer menjadi dense vectors
   - **LSTM Layers**: Memproses sequence dengan mempertahankan memory
   - **Dense Layer**: Output layer dengan softmax activation

3. **Loss Function**: `sparse_categorical_crossentropy`
4. **Optimizer**: Adam
5. **Metrics**: Accuracy

**Rumus LSTM:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)  # Hidden state
```

### 💻 Implementasi Kode

```python
# Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(doc_list)
sequences = tokenizer.texts_to_sequences(doc_list)
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# Encode labels
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_idx[label] for label in labels])

# Build Model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(len(unique_labels), activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

# Prediction
prediction = model.predict(test_padded)
predicted_label = unique_labels[np.argmax(prediction[0])]
```

**Library yang digunakan:**
- `tensorflow.keras`: Untuk building dan training model
- `tensorflow.keras.layers`: LSTM, Dense, Embedding
- `tensorflow.keras.preprocessing.text`: Tokenizer
- `transformers.pipeline`: Untuk text generation (GPT-2)

### 📊 Visualisasi

1. **Training History - Accuracy**
   - Line chart menampilkan accuracy training vs validation
   - Sumbu X: Epoch (1-10)
   - Sumbu Y: Accuracy (0-1)
   - Dua garis: Training (biru) dan Validation (orange)

2. **Training History - Loss**
   - Line chart menampilkan loss training vs validation
   - Sumbu X: Epoch (1-10)
   - Sumbu Y: Loss
   - Dua garis: Training (biru) dan Validation (orange)

3. **Metrics Display**
   - Training Accuracy: nilai akhir
   - Validation Accuracy: nilai akhir

### ⚙️ Fitur Menu

#### A. Text Classification
1. **Input Format**:
   - Manual: `teks|label` (satu per baris)
   - CSV: File dengan kolom 'text' dan 'label'

2. **Fitur**:
   - ✅ Training model LSTM untuk klasifikasi
   - ✅ Visualisasi training history
   - ✅ Prediksi teks baru
   - ✅ Menampilkan probabilitas semua kategori

3. **Parameter**:
   - Epochs: 10 (fixed)
   - Batch size: 32 (fixed)
   - Train/Test split: 80/20

#### B. Text Generation
1. **Input**:
   - Seed text (teks awal)
   - Panjang teks yang di-generate: 10-100

2. **Fitur**:
   - ✅ Generate teks menggunakan GPT-2
   - ✅ Highlight seed text dalam hasil
   - ✅ Model pre-trained (tidak perlu training)

### 📝 Contoh Hasilnya

#### Text Classification

**Input:**
```
Saya suka belajar machine learning|positif
Deep learning sangat menarik|positif
Saya tidak suka matematika|negatif
Python adalah bahasa yang mudah|positif
Pemrograman itu sulit|negatif
```

**Output:**

**Hasil Pelatihan:**
- Training Accuracy: **0.9500**
- Validation Accuracy: **0.8000**

**Prediksi Teks Baru:**
- Input: "Saya suka belajar AI"
- Prediksi: **positif** (Confidence: 0.9234)

**Probabilitas:**
| Label | Probabilitas |
|-------|-------------|
| positif | 0.9234 |
| negatif | 0.0766 |

#### Text Generation

**Input:**
- Seed text: "Machine learning adalah"
- Max length: 50

**Output:**
```
Machine learning adalah cabang dari artificial intelligence yang 
memungkinkan sistem untuk belajar dari data tanpa diprogram secara 
eksplisit. Teknologi ini telah mengubah cara kita memproses informasi 
dan membuat keputusan.
```

---

## 🔄 3. SEQUENCE TO SEQUENCE (SEQ2SEQ)

### 📋 Fungsi Menu
Menu ini menggunakan model Seq2Seq berbasis Transformer untuk tiga task:
1. **Translation**: Menerjemahkan teks antar bahasa
2. **Summarization**: Membuat ringkasan dari teks panjang
3. **Question Answering**: Menjawab pertanyaan berdasarkan konteks

**Kegunaan:**
- Penerjemahan otomatis
- Ringkasan dokumen panjang
- Sistem tanya jawab otomatis

### 🧮 Algoritma

#### Transformer Architecture (Encoder-Decoder)
1. **Encoder**:
   - Multi-head self-attention
   - Position-wise feed-forward networks
   - Residual connections & layer normalization

2. **Decoder**:
   - Masked multi-head self-attention
   - Multi-head encoder-decoder attention
   - Position-wise feed-forward networks

3. **Attention Mechanism**:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) V
   ```

**Model yang Digunakan:**
- **Translation**: Helsinki-NLP/opus-mt (MarianMT)
- **Summarization**: facebook/bart-large-cnn (BART)
- **Question Answering**: distilbert-base-uncased (DistilBERT)

### 💻 Implementasi Kode

```python
# Translation
translator = pipeline("translation", 
                     model="Helsinki-NLP/opus-mt-en-id", 
                     device=-1)
result = translator(text_to_translate)
translated_text = result[0]['translation_text']

# Summarization
summarizer = pipeline("summarization", 
                      model="facebook/bart-large-cnn", 
                      device=-1)
result = summarizer(text, max_length=50, min_length=20)
summary = result[0]['summary_text']

# Question Answering
qa_pipeline = pipeline("question-answering", device=-1)
result = qa_pipeline(question=question, context=context)
answer = result['answer']
confidence = result['score']
```

**Library yang digunakan:**
- `transformers.pipeline`: High-level API untuk berbagai NLP tasks
- Model pre-trained dari Hugging Face

### 📊 Visualisasi

1. **Translation**:
   - Side-by-side comparison (Bahasa Sumber vs Bahasa Target)
   - Metrics: Panjang teks (karakter/kata)

2. **Summarization**:
   - Metrics comparison:
     - Panjang teks asli (kata)
     - Panjang ringkasan (kata)
     - Rasio kompresi

3. **Question Answering**:
   - Highlight jawaban dalam konteks (yellow background)
   - Confidence score (0-1)

### ⚙️ Fitur Menu

#### A. Translation
1. **Bahasa yang Didukung**:
   - English ↔ Indonesian
   - English ↔ French
   - French ↔ English
   - Default: English → Indonesian

2. **Fitur**:
   - ✅ Pilih bahasa sumber dan target
   - ✅ Terjemahan real-time
   - ✅ Side-by-side display

#### B. Summarization
1. **Parameter**:
   - Panjang maksimal ringkasan: 30-150 kata
   - Panjang minimal ringkasan: 10-50 kata

2. **Fitur**:
   - ✅ Ringkasan otomatis dari teks panjang
   - ✅ Kontrol panjang ringkasan
   - ✅ Perbandingan panjang teks

#### C. Question Answering
1. **Input**:
   - Konteks (teks referensi)
   - Pertanyaan

2. **Fitur**:
   - ✅ Ekstraksi jawaban dari konteks
   - ✅ Confidence score
   - ✅ Highlight jawaban dalam konteks

### 📝 Contoh Hasilnya

#### Translation

**Input:**
- Bahasa Sumber: English
- Bahasa Target: Indonesian
- Teks: "Hello, how are you?"

**Output:**
- **English:** Hello, how are you?
- **Indonesian:** Halo, apa kabar?

#### Summarization

**Input:**
```
Machine learning is a subset of artificial intelligence that focuses 
on the development of algorithms and statistical models that enable 
computer systems to improve their performance on a specific task 
through experience...
```

**Output:**
- **Ringkasan:** "Machine learning is a subset of AI that enables systems to improve through experience using algorithms and neural networks."

- **Panjang Teks Asli:** 85 kata
- **Panjang Ringkasan:** 18 kata
- **Rasio Kompresi:** 21.2%

#### Question Answering

**Input:**
- Konteks: "Machine learning is a method of data analysis that automates analytical model building..."
- Pertanyaan: "What is machine learning?"

**Output:**
- **Jawaban:** "a method of data analysis that automates analytical model building"
- **Confidence Score:** 0.9234
- **Highlight:** Jawaban ditandai dengan background kuning dalam konteks

---

## ⚡ 4. TRANSFORMERS

### 📋 Fungsi Menu
Menu ini menggunakan model Transformer pre-trained untuk berbagai task NLP:
1. **Sentiment Analysis**: Menganalisis sentimen (positif/negatif)
2. **Named Entity Recognition (NER)**: Mengidentifikasi entitas (nama, lokasi, organisasi, dll)
3. **Text Classification**: Mengklasifikasikan teks ke kategori
4. **Zero-Shot Classification**: Klasifikasi tanpa training, hanya dengan label

**Kegunaan:**
- Analisis sentimen review, tweet, dll
- Ekstraksi informasi (nama, lokasi, tanggal, dll)
- Klasifikasi dokumen
- Klasifikasi fleksibel tanpa training

### 🧮 Algoritma

#### Transformer Architecture
1. **Self-Attention Mechanism**:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) V
   ```

2. **Multi-Head Attention**:
   - Multiple attention heads untuk menangkap berbagai aspek
   - Concatenate hasil dari semua heads

3. **Model yang Digunakan**:
   - **Sentiment Analysis**: distilbert-base-uncased-finetuned-sst-2-english
   - **NER**: dbmdz/bert-large-cased-finetuned-conll03-english
   - **Text Classification**: facebook/bart-large-mnli (zero-shot)
   - **Zero-Shot**: facebook/bart-large-mnli

### 💻 Implementasi Kode

```python
# Sentiment Analysis
classifier = pipeline("sentiment-analysis", device=-1)
result = classifier(text)
# Output: {'label': 'POSITIVE', 'score': 0.9998}

# Named Entity Recognition
ner_pipeline = pipeline("ner", 
                       aggregation_strategy="simple", 
                       device=-1)
entities = ner_pipeline(text)
# Output: [{'word': 'Barack Obama', 'entity_group': 'PER', 'score': 0.99}]

# Zero-Shot Classification
classifier = pipeline("zero-shot-classification", device=-1)
result = classifier(text, candidate_labels=["positive", "negative"])
# Output: {'labels': ['positive', 'negative'], 'scores': [0.95, 0.05]}
```

**Library yang digunakan:**
- `transformers.pipeline`: High-level API
- Model pre-trained dari Hugging Face Model Hub

### 📊 Visualisasi

1. **Sentiment Analysis**:
   - Bar chart distribusi sentiment
   - Warna: Hijau (positive), Merah (negative), Biru (neutral)
   - Sumbu X: Sentiment label
   - Sumbu Y: Jumlah dokumen

2. **Named Entity Recognition**:
   - Horizontal bar chart distribusi entity types
   - Warna: Coral
   - Sumbu X: Jumlah
   - Sumbu Y: Entity type (PER, ORG, LOC, dll)

3. **Zero-Shot Classification**:
   - Horizontal bar chart scores untuk semua label
   - Warna: Steel blue
   - Sumbu X: Score (0-1)
   - Sumbu Y: Label

### ⚙️ Fitur Menu

#### A. Sentiment Analysis
1. **Input**:
   - Manual: Teks langsung
   - File: CSV/TXT dengan multiple dokumen

2. **Fitur**:
   - ✅ Analisis sentimen untuk setiap dokumen
   - ✅ Score confidence (0-1)
   - ✅ Visualisasi distribusi sentiment
   - ✅ Export hasil ke CSV/Excel

3. **Output Labels**:
   - POSITIVE / NEGATIVE
   - Kadang: NEUTRAL

#### B. Named Entity Recognition (NER)
1. **Entity Types**:
   - PER (Person): Nama orang
   - ORG (Organization): Organisasi
   - LOC (Location): Lokasi
   - MISC (Miscellaneous): Lainnya

2. **Fitur**:
   - ✅ Ekstraksi semua entitas dari teks
   - ✅ Label entity type
   - ✅ Confidence score
   - ✅ Visualisasi distribusi entity types
   - ✅ Export hasil ke CSV/Excel

#### C. Text Classification
1. **Input**:
   - Teks untuk diklasifikasikan
   - Kategori (custom, dipisahkan koma)

2. **Fitur**:
   - ✅ Klasifikasi ke kategori custom
   - ✅ Menampilkan probabilitas semua kategori
   - ✅ Export hasil

#### D. Zero-Shot Classification
1. **Input**:
   - Teks untuk diklasifikasikan
   - Kategori (custom, dipisahkan koma)

2. **Fitur**:
   - ✅ Klasifikasi tanpa training
   - ✅ Label fleksibel (user-defined)
   - ✅ Visualisasi scores
   - ✅ Tidak perlu dataset training

### 📝 Contoh Hasilnya

#### Sentiment Analysis

**Input:**
```
I love this product! It's amazing.
This is terrible, I hate it.
The product is okay, nothing special.
```

**Output:**

| Teks | Label | Score |
|------|-------|-------|
| I love this product! It's amazing. | POSITIVE | 0.9998 |
| This is terrible, I hate it. | NEGATIVE | 0.9876 |
| The product is okay, nothing special. | NEUTRAL | 0.6543 |

**Visualisasi:**
- POSITIVE: 1 dokumen
- NEGATIVE: 1 dokumen
- NEUTRAL: 1 dokumen

#### Named Entity Recognition

**Input:**
```
Barack Obama was born in Hawaii. He worked at the White House.
```

**Output:**

| Entity | Label | Score |
|--------|-------|-------|
| Barack Obama | PER | 0.9987 |
| Hawaii | LOC | 0.9876 |
| White House | ORG | 0.9234 |

**Visualisasi:**
- PER: 1
- LOC: 1
- ORG: 1

#### Zero-Shot Classification

**Input:**
- Teks: "This is a great movie with excellent acting."
- Kategori: "positive, negative, neutral"

**Output:**
- **Prediksi:** positive (Score: 0.9234)

**Probabilitas Semua Kategori:**
| Label | Score |
|-------|-------|
| positive | 0.9234 |
| neutral | 0.0654 |
| negative | 0.0112 |

---

## 📊 Ringkasan Perbandingan Fitur

| Fitur | Algoritma Utama | Library | Input | Output |
|-------|----------------|---------|-------|--------|
| **Word Vector** | Word2Vec (CBOW) | Gensim | Dokumen teks | Vektor kata, similarity |
| **RNN** | LSTM | TensorFlow/Keras | Teks + Label | Model klasifikasi, prediksi |
| **Seq2Seq** | Transformer | Transformers | Teks (translation/summary/QA) | Terjemahan/Ringkasan/Jawaban |
| **Transformers** | Pre-trained Transformer | Transformers | Teks | Sentiment/NER/Classification |

---

## 🔧 Teknologi yang Digunakan

1. **Word Vector Representations**:
   - Gensim (Word2Vec)
   - NLTK (tokenization)

2. **RNN**:
   - TensorFlow 2.20
   - Keras 3.13
   - Transformers (untuk text generation)

3. **Seq2Seq**:
   - Transformers 4.57
   - PyTorch 2.9
   - Hugging Face Model Hub

4. **Transformers**:
   - Transformers 4.57
   - PyTorch 2.9
   - Pre-trained models dari Hugging Face

---

## 📝 Catatan Penting

1. **Model Pre-trained**: Seq2Seq dan Transformers menggunakan model pre-trained yang akan diunduh otomatis saat pertama kali digunakan (butuh koneksi internet).

2. **Resource Requirements**:
   - Word Vector: Ringan, cepat
   - RNN: Sedang, perlu training
   - Seq2Seq: Berat, model besar (1-3 GB)
   - Transformers: Berat, model besar (500 MB - 2 GB)

3. **Waktu Proses**:
   - Word Vector: Beberapa detik
   - RNN Training: 1-5 menit (tergantung data)
   - Seq2Seq: 10-30 detik (pertama kali lebih lama karena download)
   - Transformers: 5-15 detik (pertama kali lebih lama karena download)

4. **Bahasa yang Didukung**:
   - Word Vector: Semua bahasa (tergantung data input)
   - RNN: Semua bahasa (tergantung data training)
   - Seq2Seq: English, Indonesian, French, German, Spanish (tergantung model)
   - Transformers: English (default), bisa fine-tune untuk bahasa lain

---

**Dokumentasi ini dibuat untuk membantu memahami fitur-fitur baru dalam aplikasi NLP - One for All.**

