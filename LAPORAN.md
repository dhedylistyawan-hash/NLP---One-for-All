# 📊 LAPORAN SISTEM NLP - ONE FOR ALL

## DAFTAR ISI
1. [Pendahuluan](#pendahuluan)
2. [Cara Membuat Sistem](#cara-membuat-sistem)
3. [Penjelasan Setiap Menu](#penjelasan-setiap-menu)
4. [Hasil dan Fitur](#hasil-dan-fitur)
5. [Kesimpulan](#kesimpulan)

---

## PENDAHULUAN

### 1.1 Deskripsi Sistem
**NLP - One for All** adalah aplikasi web interaktif berbasis Streamlit yang menyediakan berbagai tools untuk Natural Language Processing (NLP). Aplikasi ini dirancang untuk membantu pengguna dalam menganalisis dan memproses teks dengan berbagai metode yang populer dalam bidang NLP.

### 1.2 Fitur Utama
- **Feature Extraction (TF-IDF)**: Ekstraksi fitur menggunakan Term Frequency-Inverse Document Frequency
- **Inverted Index**: Pembuatan struktur data inverted index untuk pencarian teks
- **Model LDA**: Analisis topik menggunakan Latent Dirichlet Allocation
- **Model BERTopic**: Analisis topik menggunakan BERTopic dengan transformer embedding

### 1.3 Teknologi yang Digunakan
- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python 3.8+
- **Libraries**: Pandas, Scikit-learn, Gensim, BERTopic, NLTK, Plotly, Matplotlib, Seaborn, WordCloud

---

## CARA MEMBUAT SISTEM

### 2.1 Persiapan Environment

#### 2.1.1 Instalasi Python
```bash
# Pastikan Python 3.8 atau lebih tinggi terinstall
python --version
```

#### 2.1.2 Membuat Virtual Environment (Opsional)
```bash
# Buat virtual environment
python -m venv .venv

# Aktifkan virtual environment
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
```

### 2.2 Instalasi Dependencies

#### 2.2.1 File requirements.txt
File `requirements.txt` berisi semua library yang diperlukan:

```
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
gensim>=4.3.0
bertopic>=0.15.0
plotly>=5.17.0
sentence-transformers>=2.2.0
tf-keras>=2.20.0
wordcloud>=1.9.0
seaborn>=0.12.0
matplotlib>=3.7.0
pyldavis>=3.4.0
openpyxl>=3.1.0
pillow>=10.0.0
```

#### 2.2.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 2.3 Menjalankan Aplikasi

```bash
# Jalankan aplikasi Streamlit
streamlit run app.py

# Atau menggunakan Python module
python -m streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

### 2.4 Struktur File Proyek

```
NLP/
├── app.py                 # File utama aplikasi
├── requirements.txt       # Daftar dependencies
├── README.md             # Dokumentasi dasar
├── LAPORAN.md            # Laporan lengkap (file ini)
└── pyrightconfig.json    # Konfigurasi type checker
```

---

## PENJELASAN SETIAP MENU

### 3.1 MENU 1: FEATURE EXTRACTION (TF-IDF)

#### 3.1.1 Deskripsi Fungsi
Menu ini melakukan ekstraksi fitur dari dokumen menggunakan metode **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF mengukur seberapa penting sebuah kata dalam dokumen relatif terhadap kumpulan dokumen.

#### 3.1.2 Algoritma TF-IDF
- **TF (Term Frequency)**: Frekuensi kemunculan kata dalam dokumen
- **IDF (Inverse Document Frequency)**: Logaritma dari rasio jumlah dokumen total dibagi jumlah dokumen yang mengandung kata tersebut
- **TF-IDF Score**: `TF × IDF`

#### 3.1.3 Code dan Penjelasan

```python
# Import library yang diperlukan
from sklearn.feature_extraction.text import TfidfVectorizer

# Membuat TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit dan transform dokumen menjadi matriks TF-IDF
tfidf_matrix = vectorizer.fit_transform(doc_list)

# Konversi ke array dan DataFrame untuk visualisasi
tfidf_array = tfidf_matrix.toarray()
df = pd.DataFrame(
    tfidf_array,
    columns=vectorizer.get_feature_names_out(),
    index=[f'Dokumen {i+1}' for i in range(len(doc_list))]
)
```

**Penjelasan Code:**
1. `TfidfVectorizer()`: Membuat objek vectorizer untuk menghitung TF-IDF
2. `fit_transform()`: Melatih model dan mengkonversi dokumen menjadi matriks sparse
3. `toarray()`: Mengkonversi matriks sparse menjadi array numpy
4. `DataFrame`: Menyimpan hasil dalam format tabel untuk visualisasi

#### 3.1.4 Fitur yang Tersedia
- ✅ Upload file CSV/TXT atau input manual
- ✅ Visualisasi Heatmap TF-IDF
- ✅ Word Cloud dari kata-kata penting
- ✅ Grafik Top 10 kata dengan skor tertinggi
- ✅ Vocabulary dengan total skor TF-IDF
- ✅ Export hasil ke CSV/Excel

#### 3.1.5 Contoh Hasil
- **Matriks TF-IDF**: Tabel yang menampilkan skor TF-IDF untuk setiap kata di setiap dokumen
- **Heatmap**: Visualisasi warna yang menunjukkan distribusi skor TF-IDF
- **Word Cloud**: Visualisasi kata-kata dengan ukuran proporsional terhadap skor TF-IDF
- **Bar Chart**: Grafik horizontal menampilkan 10 kata dengan skor tertinggi

---

### 3.2 MENU 2: INVERTED INDEX

#### 3.2.1 Deskripsi Fungsi
Menu ini membuat **Inverted Index**, struktur data yang memetakan setiap kata ke daftar dokumen yang mengandung kata tersebut. Struktur ini sangat efisien untuk pencarian teks.

#### 3.2.2 Algoritma Inverted Index
1. Tokenisasi: Memisahkan teks menjadi kata-kata (tokens)
2. Normalisasi: Mengubah ke lowercase dan menghapus tanda baca
3. Indexing: Membuat mapping dari kata ke dokumen yang mengandungnya
4. Sorting: Mengurutkan ID dokumen untuk setiap kata

#### 3.2.3 Code dan Penjelasan

```python
def create_inverted_index(docs):
    """Membuat inverted index dari list dokumen"""
    inverted_index = {}
    word_freq = {}  # Frekuensi kata per dokumen
    
    # Iterasi setiap dokumen
    for doc_id, text in enumerate(docs, start=1):
        # Tokenisasi sederhana
        tokens = text.lower().split()
        
        # Iterasi setiap token
        for token in tokens:
            # Bersihkan token (hapus tanda baca)
            token = ''.join(c for c in token if c.isalnum())
            
            if token:  # Pastikan token tidak kosong
                # Inisialisasi jika belum ada
                if token not in inverted_index:
                    inverted_index[token] = []
                    word_freq[token] = {}
                
                # Tambahkan doc_id jika belum ada
                if doc_id not in inverted_index[token]:
                    inverted_index[token].append(doc_id)
                    word_freq[token][doc_id] = 0
                
                # Hitung frekuensi
                word_freq[token][doc_id] = word_freq[token].get(doc_id, 0) + 1
    
    # Sortir list doc_id untuk setiap kata
    for token in inverted_index:
        inverted_index[token].sort()
    
    return inverted_index, word_freq
```

**Penjelasan Code:**
1. **Tokenisasi**: Memisahkan teks menjadi kata-kata menggunakan `split()`
2. **Normalisasi**: Mengubah ke lowercase dan menghapus karakter non-alfanumerik
3. **Indexing**: Menyimpan mapping `kata → [doc_id1, doc_id2, ...]`
4. **Frekuensi**: Menghitung berapa kali kata muncul di setiap dokumen

#### 3.2.4 Fitur Pencarian

**a. Pencarian Kata Tunggal:**
```python
search_word = search_query.lower().strip()
if search_word in index_result:
    doc_ids = index_result[search_word]
    # Tampilkan dokumen yang mengandung kata tersebut
```

**b. Pencarian Multi-kata (AND):**
```python
# Mencari dokumen yang mengandung SEMUA kata
words = multi_search.split()
found_docs = set(index_result[words[0]])
for word in words[1:]:
    found_docs &= set(index_result[word])  # Intersection
```

**c. Pencarian Multi-kata (OR):**
```python
# Mencari dokumen yang mengandung SALAH SATU kata
found_docs = set()
for word in words:
    if word in index_result:
        found_docs.update(index_result[word])  # Union
```

#### 3.2.5 Fitur yang Tersedia
- ✅ Upload file CSV/TXT atau input manual
- ✅ Pencarian kata tunggal dengan highlighting
- ✅ Pencarian multi-kata (AND/OR)
- ✅ Statistik (total kata unik, dokumen, frekuensi)
- ✅ Visualisasi grafik kata terpopuler
- ✅ Export hasil ke CSV/Excel

#### 3.2.6 Contoh Hasil
- **Inverted Index Table**: Tabel menampilkan setiap kata dengan daftar dokumen yang mengandungnya
- **Hasil Pencarian**: Daftar dokumen yang ditemukan dengan highlighting kata yang dicari
- **Statistik**: Metrics menampilkan total kata unik, dokumen, dan frekuensi
- **Bar Chart**: Grafik top 15 kata dengan frekuensi tertinggi

---

### 3.3 MENU 3: MODEL LDA (LATENT DIRICHLET ALLOCATION)

#### 3.3.1 Deskripsi Fungsi
Menu ini melatih model **LDA (Latent Dirichlet Allocation)** untuk menemukan topik-topik tersembunyi dalam kumpulan dokumen. LDA adalah model probabilistik yang mengasumsikan bahwa setiap dokumen adalah campuran dari beberapa topik.

#### 3.3.2 Algoritma LDA
1. **Preprocessing**: Tokenisasi, lowercase, hapus stopwords, filter karakter
2. **Dictionary & Corpus**: Membuat vocabulary dan Bag of Words representation
3. **Training**: Melatih model LDA dengan jumlah topik tertentu
4. **Topic Extraction**: Mengekstrak kata-kata kunci untuk setiap topik

#### 3.3.3 Code dan Penjelasan

```python
# Preprocessing
def preprocess(text):
    """Preprocessing teks: tokenisasi, lowercase, hapus stopwords"""
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens 
             if token.isalpha() 
             and token not in stop_words 
             and len(token) > 2]
    return tokens

# Preprocess semua dokumen
processed_texts = [preprocess(doc) for doc in doc_list]

# Buat dictionary dan corpus
from gensim import corpora
from gensim.models import LdaModel

dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Latih model LDA
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,  # Jumlah topik yang diinginkan
    random_state=42,
    passes=15,              # Jumlah iterasi
    alpha='auto',           # Prior untuk distribusi topik
    per_word_topics=True
)

# Ambil topik
topics = lda_model.print_topics(num_words=10)
```

**Penjelasan Code:**
1. **Preprocessing**: Membersihkan teks dari stopwords dan karakter tidak penting
2. **Dictionary**: Membuat mapping kata → ID untuk efisiensi
3. **Corpus (Bag of Words)**: Representasi dokumen sebagai vektor frekuensi kata
4. **LDA Training**: Melatih model untuk menemukan distribusi topik
5. **Topic Extraction**: Mengambil kata-kata dengan probabilitas tertinggi untuk setiap topik

#### 3.3.4 Visualisasi

**a. Visualisasi Interaktif (pyLDAvis):**
```python
import pyLDAvis
import pyLDAvis.gensim_models

vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
html_string = pyLDAvis.prepared_data_to_html(vis)
# Tampilkan di browser
```

**b. Word Cloud per Topik:**
```python
# Untuk setiap topik, buat word cloud dari kata-kata kunci
for topic in topics:
    words = extract_words_from_topic(topic)
    wordcloud = WordCloud().generate(' '.join(words))
```

**c. Distribusi Topik per Dokumen:**
```python
# Hitung distribusi topik untuk setiap dokumen
doc_topics = []
for doc_bow in corpus:
    topic_dist = lda_model.get_document_topics(doc_bow)
    doc_topics.append(topic_dist)
```

#### 3.3.5 Fitur yang Tersedia
- ✅ Upload file CSV/TXT atau input manual
- ✅ Pilihan bahasa (Indonesia/Inggris)
- ✅ Slider untuk jumlah topik (2-10)
- ✅ Visualisasi interaktif pyLDAvis
- ✅ Word Cloud per topik
- ✅ Grafik distribusi topik per dokumen
- ✅ Export hasil ke CSV/Excel

#### 3.3.6 Contoh Hasil
- **Topik yang Ditemukan**: Daftar topik dengan 10 kata kunci teratas
- **Intertopic Distance Map**: Visualisasi jarak antar topik secara interaktif
- **Word Cloud**: Visualisasi kata-kata penting untuk setiap topik
- **Distribusi Topik**: Grafik stacked bar menunjukkan distribusi topik per dokumen

---

### 3.4 MENU 4: MODEL BERTopic

#### 3.4.1 Deskripsi Fungsi
Menu ini menggunakan **BERTopic**, library modern untuk analisis topik yang menggabungkan embedding transformer (BERT) dengan teknik clustering (UMAP + HDBSCAN). BERTopic lebih canggih daripada LDA karena menggunakan semantic understanding dari transformer.

#### 3.4.2 Algoritma BERTopic
1. **Embedding**: Menggunakan sentence transformer untuk mengubah dokumen menjadi vektor
2. **Dimensionality Reduction**: Menggunakan UMAP untuk mengurangi dimensi
3. **Clustering**: Menggunakan HDBSCAN untuk menemukan cluster (topik)
4. **Topic Representation**: Mengekstrak kata-kata representatif untuk setiap topik

#### 3.4.3 Code dan Penjelasan

```python
from bertopic import BERTopic

# Tentukan model embedding berdasarkan bahasa
if bahasa == 'Inggris':
    embedding_model = 'all-MiniLM-L6-v2'
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=False,
        calculate_probabilities=True
    )
else:  # Indonesia/Multilingual
    embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    topic_model = BERTopic(
        language='multilingual',
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        verbose=False,
        calculate_probabilities=True
    )

# Latih model
topics, probs = topic_model.fit_transform(doc_list)

# Ambil informasi topik
topic_info = topic_model.get_topic_info()
```

**Penjelasan Code:**
1. **Embedding Model**: 
   - Inggris: `all-MiniLM-L6-v2` (model ringan dan cepat)
   - Multilingual: `paraphrase-multilingual-MiniLM-L12-v2` (mendukung banyak bahasa)
2. **min_topic_size**: Ukuran minimum cluster untuk dianggap sebagai topik
3. **fit_transform()**: Melatih model dan mengembalikan topik untuk setiap dokumen
4. **get_topic_info()**: Mendapatkan informasi statistik tentang topik

#### 3.4.4 Visualisasi

**a. Visualisasi Jarak Antar Topik:**
```python
# Memerlukan minimal 2 topik (selain outlier -1)
if len(topic_ids) >= 2:
    fig = topic_model.visualize_topics()
    st.plotly_chart(fig, use_container_width=True)
```

**b. Bar Chart Topik:**
```python
fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
st.plotly_chart(fig_barchart, use_container_width=True)
```

#### 3.4.5 Fitur yang Tersedia
- ✅ Upload file CSV/TXT atau input manual
- ✅ Pilihan bahasa (Indonesia/Inggris)
- ✅ Auto-adjustment untuk dataset kecil
- ✅ Visualisasi jarak antar topik (interaktif)
- ✅ Bar chart kata kunci per topik
- ✅ Export hasil ke CSV/Excel

#### 3.4.6 Contoh Hasil
- **Info Topik**: Tabel dengan informasi setiap topik (count, name, representation)
- **Kata Kunci per Topik**: Daftar kata-kata kunci untuk setiap topik
- **Visualisasi Interaktif**: Plot interaktif menunjukkan hubungan antar topik
- **Bar Chart**: Grafik batang menampilkan kata-kata penting per topik

---

## HASIL DAN FITUR

### 4.1 Fitur Umum yang Tersedia

#### 4.1.1 Upload File
- **Format yang didukung**: CSV, TXT
- **Auto-detection**: Untuk CSV, sistem otomatis mendeteksi kolom teks
- **Preview**: Menampilkan preview dokumen setelah upload

#### 4.1.2 Export Hasil
- **Format**: CSV, Excel (.xlsx)
- **Konten**: Semua hasil analisis dapat diekspor
- **Otomatis**: Link download muncul setelah proses selesai

#### 4.1.3 Visualisasi
- **Heatmap**: Visualisasi warna untuk matriks data
- **Word Cloud**: Visualisasi kata-kata dengan ukuran proporsional
- **Bar Chart**: Grafik batang horizontal/vertikal
- **Interactive Charts**: Plotly charts untuk visualisasi interaktif
- **Stacked Bar**: Untuk distribusi data

### 4.2 Hasil yang Dihasilkan

#### 4.2.1 TF-IDF
- Matriks TF-IDF (dokumen × kata)
- Vocabulary dengan total skor
- Visualisasi heatmap, word cloud, dan bar chart

#### 4.2.2 Inverted Index
- Struktur inverted index lengkap
- Hasil pencarian dengan highlighting
- Statistik lengkap tentang corpus

#### 4.2.3 LDA
- Topik dengan kata kunci
- Distribusi topik per dokumen
- Visualisasi interaktif pyLDAvis

#### 4.2.4 BERTopic
- Topik dengan representasi semantik
- Probabilitas topik per dokumen
- Visualisasi jarak dan bar chart

---

## KESIMPULAN

### 5.1 Pencapaian
Sistem **NLP - One for All** berhasil dibuat dengan fitur-fitur berikut:
1. ✅ 4 menu utama untuk analisis NLP
2. ✅ Upload file (CSV/TXT) untuk semua menu
3. ✅ Visualisasi yang komprehensif untuk setiap fitur
4. ✅ Export hasil ke CSV/Excel
5. ✅ Fitur pencarian untuk Inverted Index
6. ✅ Dukungan multi-bahasa (Indonesia/Inggris)
7. ✅ Error handling yang baik
8. ✅ UI yang user-friendly

### 5.2 Teknologi yang Berhasil Diimplementasikan
- **Streamlit**: Framework web untuk UI
- **Scikit-learn**: TF-IDF vectorization
- **Gensim**: LDA topic modeling
- **BERTopic**: Advanced topic modeling dengan transformer
- **Plotly**: Visualisasi interaktif
- **WordCloud**: Visualisasi kata-kata
- **Seaborn & Matplotlib**: Visualisasi statis

### 5.3 Keunggulan Sistem
1. **Mudah Digunakan**: Interface yang intuitif, tidak perlu pengetahuan programming
2. **Fleksibel**: Mendukung input manual dan upload file
3. **Komprehensif**: Mencakup berbagai teknik NLP populer
4. **Visualisasi Kaya**: Berbagai jenis visualisasi untuk setiap fitur
5. **Export Ready**: Hasil dapat diekspor untuk analisis lebih lanjut

### 5.4 Saran Pengembangan
1. Tambahkan support untuk format file lain (PDF, DOCX)
2. Implementasi sentiment analysis
3. Text summarization
4. Named Entity Recognition (NER)
5. Text classification
6. Integration dengan database
7. User authentication untuk multi-user

---

## LAMPIRAN

### A. Daftar Library yang Digunakan
```
streamlit>=1.28.0      # Web framework
pandas>=2.0.0          # Data manipulation
scikit-learn>=1.3.0    # Machine learning (TF-IDF)
nltk>=3.8.0            # Natural language toolkit
gensim>=4.3.0          # Topic modeling (LDA)
bertopic>=0.15.0       # Advanced topic modeling
plotly>=5.17.0         # Interactive visualization
sentence-transformers>=2.2.0  # Embedding models
wordcloud>=1.9.0       # Word cloud generation
seaborn>=0.12.0        # Statistical visualization
matplotlib>=3.7.0      # Plotting library
pyldavis>=3.4.0        # LDA visualization
openpyxl>=3.1.0        # Excel export
pillow>=10.0.0         # Image processing
```

### B. Contoh Penggunaan

#### B.1 TF-IDF
```python
# Input: List dokumen
documents = [
    "Saya suka belajar data science.",
    "Python adalah bahasa pemrograman populer.",
    "Analisis data sangat menarik."
]

# Output: Matriks TF-IDF dengan skor untuk setiap kata
```

#### B.2 Inverted Index
```python
# Input: List dokumen
# Output: Dictionary dengan struktur:
{
    "kucing": [1, 2, 5],    # Dokumen 1, 2, 5 mengandung "kucing"
    "anjing": [2, 5],       # Dokumen 2, 5 mengandung "anjing"
    ...
}
```

#### B.3 LDA
```python
# Input: List dokumen, jumlah topik
# Output: 
# - Topik dengan kata kunci
# - Distribusi topik per dokumen
```

#### B.4 BERTopic
```python
# Input: List dokumen
# Output:
# - Topik dengan representasi semantik
# - Probabilitas topik untuk setiap dokumen
```

---

**Dokumen ini dibuat untuk keperluan dokumentasi sistem NLP - One for All**

*Terakhir diperbarui: 2024*

