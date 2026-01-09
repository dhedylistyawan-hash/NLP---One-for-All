import os
# Set environment variable untuk kompatibilitas Keras 3 dengan Transformers
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora  # type: ignore
from gensim.models import LdaModel, Word2Vec  # type: ignore
try:
    from gensim.models import FastText  # type: ignore
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io
import base64
from wordcloud import WordCloud  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import pyLDAvis  # type: ignore
    import pyLDAvis.gensim_models  # type: ignore
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False

try:
    from bertopic import BERTopic  # type: ignore
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Import untuk fitur baru
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel  # type: ignore
    from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Embedding, SimpleRNN, GRU  # type: ignore
    from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Download NLTK data yang diperlukan
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Download punkt_tab untuk NLTK versi terbaru
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback jika punkt_tab tidak tersedia
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ===========================================
# HELPER FUNCTIONS
# ===========================================

def load_uploaded_file(uploaded_file):
    """Load file yang diupload (CSV atau TXT)"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Cari kolom yang berisi teks
            text_col = None
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
                    text_col = col
                    break
            if text_col:
                doc_list = df[text_col].dropna().astype(str).tolist()
                return doc_list, f"File CSV dengan kolom '{text_col}'"
            else:
                # Jika tidak ada kolom yang cocok, gabungkan semua kolom teks
                text_cols = df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    doc_list = df[text_cols[0]].dropna().astype(str).tolist()
                    return doc_list, f"File CSV dengan kolom '{text_cols[0]}'"
                else:
                    return None, "Tidak ada kolom teks yang ditemukan"
        else:  # TXT
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            doc_list = [line.strip() for line in content.split('\n') if line.strip()]
            return doc_list, "File TXT"
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_download_link(df, filename="hasil.csv", file_format="csv"):
    """Buat link download untuk DataFrame"""
    if file_format == "csv":
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    else:  # Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=True)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Download Excel</a>'
    return href

def create_wordcloud(text_data, title="Word Cloud"):
    """Buat word cloud dari data teks"""
    try:
        # Gabungkan semua teks
        if isinstance(text_data, list):
            text = ' '.join(text_data)
        else:
            text = text_data
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=100, colormap='viridis').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# Konfigurasi halaman
st.set_page_config(
    page_title="NLP - One for All",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling modern
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 NLP - One for All</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("📚 Menu Program")
program = st.sidebar.radio(
    "Pilih Program:",
    [
        "🏠 Beranda", 
        "🔍 Feature Extraction (TF-IDF)", 
        "📑 Inverted Index", 
        "🎯 Model LDA", 
        "🤖 Model BERTopic",
        "🔤 Word Vector Representations",
        "🧠 Recurrent Neural Network (RNN)",
        "🔄 Sequence to Sequence (Seq2Seq)",
        "⚡ Transformers"
    ],
    index=0
)

# ===========================================
# PROGRAM 1: FEATURE EXTRACTION (TF-IDF)
# ===========================================
if program == "🔍 Feature Extraction (TF-IDF)":
    st.markdown('<h2 class="subheader">🔍 Feature Extraction dengan TF-IDF</h2>', unsafe_allow_html=True)
    st.write("Program ini mengekstraksi fitur dari dokumen menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency).")
    
    # Upload file atau input manual
    input_method = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload File (CSV/TXT)"], horizontal=True)
    
    doc_list = []
    input_source = ""
    
    if input_method == "📁 Upload File (CSV/TXT)":
        uploaded_file = st.file_uploader("Pilih file", type=['csv', 'txt'])
        if uploaded_file is not None:
            doc_list, input_source = load_uploaded_file(uploaded_file)
            if doc_list is None:
                st.error(f"❌ {input_source}")
            else:
                st.success(f"✅ {input_source} berhasil dimuat! Ditemukan {len(doc_list)} dokumen.")
                with st.expander("📄 Preview dokumen"):
                    for i, doc in enumerate(doc_list[:5], 1):
                        st.write(f"**Dokumen {i}:** {doc[:100]}..." if len(doc) > 100 else f"**Dokumen {i}:** {doc}")
    else:
        default_docs = "Saya suka belajar data science.\nPython adalah bahasa pemrograman populer.\nAnalisis data sangat menarik.\nMachine learning adalah bagian dari data science.\nPython digunakan untuk analisis data."
        documents = st.text_area(
            'Masukkan dokumen (satu per baris):',
            value=default_docs,
            height=200,
            help="Masukkan setiap dokumen dalam baris terpisah"
        )
        if documents.strip():
            doc_list = [doc.strip() for doc in documents.split('\n') if doc.strip()]
    
    if st.button('🔢 Hitung TF-IDF', type="primary"):
        if not doc_list or len(doc_list) < 2:
            st.warning("⚠️ Minimal diperlukan 2 dokumen untuk perhitungan TF-IDF.")
        elif len(doc_list) > 0:
            with st.spinner('⏳ Sedang menghitung TF-IDF...'):
                # Membuat TF-IDF vectorizer
                vectorizer = TfidfVectorizer()
                
                # Fit dan transform dokumen
                tfidf_matrix = vectorizer.fit_transform(doc_list)
                
                # Ambil nama fitur (vocabulary)
                feature_names = vectorizer.get_feature_names_out()
                
                # Konversi ke DataFrame
                # type: ignore[union-attr] - TfidfVectorizer.fit_transform returns sparse matrix
                tfidf_array = tfidf_matrix.toarray()  # type: ignore[union-attr]
                df = pd.DataFrame(
                    tfidf_array,
                    columns=feature_names,
                    index=[f'Dokumen {i+1}' for i in range(len(doc_list))]
                )
                
                # Tampilkan hasil
                st.markdown('<h3 class="subheader">📊 Matriks TF-IDF</h3>', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                
                # Visualisasi
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<h4 class="subheader">🔥 Heatmap TF-IDF</h4>', unsafe_allow_html=True)
                    try:
                        # Ambil top 20 kata dengan skor tertinggi
                        top_words_idx = df.sum().nlargest(min(20, len(feature_names))).index
                        df_top = df[top_words_idx]
                        
                        fig, ax = plt.subplots(figsize=(12, max(6, len(doc_list) * 0.4)))
                        sns.heatmap(df_top, annot=False, fmt='.3f', cmap='YlOrRd', 
                                   cbar_kws={'label': 'TF-IDF Score'}, ax=ax)
                        ax.set_title('Heatmap TF-IDF (Top 20 Kata)', fontsize=14, pad=15)
                        ax.set_xlabel('Kata Kunci', fontsize=12)
                        ax.set_ylabel('Dokumen', fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.info(f"Visualisasi heatmap tidak tersedia: {str(e)}")
                
                with col2:
                    st.markdown('<h4 class="subheader">☁️ Word Cloud</h4>', unsafe_allow_html=True)
                    try:
                        # Buat word cloud dari kata-kata dengan skor TF-IDF tertinggi
                        word_scores = {}
                        for word in feature_names:
                            word_scores[word] = df[word].sum()
                        
                        # Sort dan ambil top words
                        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
                        top_words_text = ' '.join([word for word, score in sorted_words[:50]])
                        
                        wordcloud_fig = create_wordcloud(top_words_text, "Word Cloud TF-IDF")
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                            plt.close()
                    except Exception as e:
                        st.info(f"Word cloud tidak tersedia: {str(e)}")
                
                # Grafik top words per dokumen
                st.markdown('<h4 class="subheader">📈 Top 10 Kata dengan Skor TF-IDF Tertinggi</h4>', unsafe_allow_html=True)
                try:
                    top_words_overall = df.sum().nlargest(10)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_words_overall.plot(kind='barh', ax=ax, color='steelblue')
                    ax.set_xlabel('Total TF-IDF Score', fontsize=12)
                    ax.set_ylabel('Kata Kunci', fontsize=12)
                    ax.set_title('Top 10 Kata dengan Total Skor TF-IDF Tertinggi', fontsize=14, pad=15)
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.info(f"Grafik tidak tersedia: {str(e)}")
                
                # Tampilkan vocabulary
                st.markdown('<h3 class="subheader">📝 Vocabulary (Kata Kunci)</h3>', unsafe_allow_html=True)
                vocab_df = pd.DataFrame({
                    'Kata Kunci': feature_names,
                    'Total TF-IDF Score': [df[word].sum() for word in feature_names]
                }).sort_values('Total TF-IDF Score', ascending=False)
                st.dataframe(vocab_df, use_container_width=True)
                
                # Export hasil
                st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.markdown(create_download_link(df, "tfidf_matrix.csv", "csv"), unsafe_allow_html=True)
                with col_exp2:
                    st.markdown(create_download_link(df, "tfidf_matrix.xlsx", "excel"), unsafe_allow_html=True)
                
                st.success(f"✅ Berhasil memproses {len(doc_list)} dokumen dengan {len(feature_names)} fitur unik!")
        else:
            st.error("❌ Silakan masukkan dokumen terlebih dahulu.")

# ===========================================
# PROGRAM 2: INVERTED INDEX
# ===========================================
elif program == "📑 Inverted Index":
    st.markdown('<h2 class="subheader">📑 Inverted Index dengan Fitur Pencarian</h2>', unsafe_allow_html=True)
    st.write("Program ini membuat inverted index untuk memetakan setiap kata ke dokumen yang mengandung kata tersebut.")
    
    # Upload file atau input manual
    input_method = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload File (CSV/TXT)"], horizontal=True)
    
    doc_list = []
    input_source = ""
    
    if input_method == "📁 Upload File (CSV/TXT)":
        uploaded_file = st.file_uploader("Pilih file", type=['csv', 'txt'])
        if uploaded_file is not None:
            doc_list, input_source = load_uploaded_file(uploaded_file)
            if doc_list is None:
                st.error(f"❌ {input_source}")
            else:
                st.success(f"✅ {input_source} berhasil dimuat! Ditemukan {len(doc_list)} dokumen.")
    else:
        default_docs_ii = "kucing berlari di taman\nanjing mengejar kucing\nburung terbang di langit\ntaman itu indah\nkucing dan anjing bermain"
        documents = st.text_area(
            'Masukkan dokumen (satu per baris):',
            value=default_docs_ii,
            height=200,
            help="Masukkan setiap dokumen dalam baris terpisah"
        )
        if documents.strip():
            doc_list = [doc.strip() for doc in documents.split('\n') if doc.strip()]
    
    # Fungsi untuk membuat inverted index
    def create_inverted_index(docs):
        """Membuat inverted index dari list dokumen"""
        inverted_index = {}
        word_freq = {}  # Frekuensi kata per dokumen
        
        # Iterasi setiap dokumen
        for doc_id, text in enumerate(docs, start=1):
            # Ubah ke lowercase dan tokenisasi sederhana
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
    
    if st.button('📚 Buat Indeks', type="primary"):
        if not doc_list:
            st.error("❌ Silakan masukkan dokumen terlebih dahulu.")
        else:
            with st.spinner('⏳ Sedang membuat inverted index...'):
                # Buat inverted index
                index_result, word_freq = create_inverted_index(doc_list)
                
                # Buat DataFrame untuk ditampilkan
                index_data = []
                for word in sorted(index_result.keys()):
                    doc_ids = index_result[word]
                    freq_info = ", ".join([f"Doc {doc_id}({word_freq[word][doc_id]}x)" for doc_id in doc_ids])
                    index_data.append({
                        'Kata': word,
                        'Dokumen': freq_info,
                        'Jumlah Dokumen': len(doc_ids),
                        'Total Frekuensi': sum(word_freq[word].values())
                    })
                
                index_df = pd.DataFrame(index_data)
                
                # Simpan ke session state untuk pencarian (penting untuk mempertahankan data setelah reload)
                st.session_state['inverted_index'] = index_result
                st.session_state['word_freq'] = word_freq
                st.session_state['doc_list'] = doc_list
                st.session_state['index_df'] = index_df
                st.session_state['index_created'] = True
    
    # Tampilkan hasil inverted index jika sudah dibuat
    if st.session_state.get('index_created', False):
        index_df = st.session_state.get('index_df')
        if index_df is not None:
            st.markdown('<h3 class="subheader">📋 Hasil Inverted Index</h3>', unsafe_allow_html=True)
            st.dataframe(index_df, use_container_width=True)
    
    # Fitur Pencarian - selalu tampilkan jika index sudah dibuat
    if st.session_state.get('index_created', False):
        # Ambil data dari session state
        search_index = st.session_state.get('inverted_index', {})
        search_word_freq = st.session_state.get('word_freq', {})
        search_doc_list = st.session_state.get('doc_list', [])
        
        st.markdown('<h3 class="subheader">🔍 Pencarian Kata</h3>', unsafe_allow_html=True)
        
        # Inisialisasi session state untuk pencarian jika belum ada
        if 'search_query' not in st.session_state:
            st.session_state['search_query'] = ''
        
        # Form pencarian dengan kolom dan button
        col_search1, col_search2 = st.columns([4, 1])
        with col_search1:
            search_query = st.text_input(
                "Masukkan kata yang ingin dicari:", 
                value=st.session_state.get('search_query', ''),
                placeholder="contoh: kucing",
                key='search_input'
            )
        with col_search2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            search_button = st.button("🔍 Cari", key='search_button', use_container_width=True)
        
        # Update session state ketika button diklik
        if search_button:
            st.session_state['search_query'] = search_query
        
        # Proses pencarian berdasarkan session state
        current_query = st.session_state.get('search_query', '')
        if current_query.strip():
            search_word = current_query.lower().strip()
            search_word = ''.join(c for c in search_word if c.isalnum())
            
            if search_word in search_index:
                doc_ids = search_index[search_word]
                st.success(f"✅ Kata '{current_query}' ditemukan di {len(doc_ids)} dokumen:")
                
                for doc_id in doc_ids:
                    if search_doc_list and doc_id <= len(search_doc_list):
                        doc_text = search_doc_list[doc_id - 1]
                        freq = search_word_freq.get(search_word, {}).get(doc_id, 0)
                        # Highlight kata yang dicari
                        highlighted_text = doc_text.replace(
                            search_word, 
                            f"**<mark style='background-color: yellow'>{search_word}</mark>**"
                        )
                        st.markdown(f"**📄 Dokumen {doc_id}** (muncul {freq}x): {highlighted_text}", unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Kata '{current_query}' tidak ditemukan dalam dokumen.")
        
        # Pencarian Multi-kata
        st.markdown('<h4 class="subheader">🔎 Pencarian Multi-kata (AND/OR)</h4>', unsafe_allow_html=True)
        
        # Inisialisasi session state untuk pencarian multi-kata jika belum ada
        if 'multi_search_query' not in st.session_state:
            st.session_state['multi_search_query'] = ''
        
        col_multi1, col_multi2 = st.columns([4, 1])
        with col_multi1:
            multi_search = st.text_input(
                "Masukkan beberapa kata (pisahkan dengan spasi untuk AND, atau koma untuk OR):", 
                value=st.session_state.get('multi_search_query', ''),
                placeholder="contoh: kucing anjing (AND) atau kucing, anjing (OR)",
                key='multi_search_input'
            )
        with col_multi2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            multi_search_button = st.button("🔍 Cari", key='multi_search_button', use_container_width=True)
        
        # Update session state ketika button diklik
        if multi_search_button:
            st.session_state['multi_search_query'] = multi_search
        
        # Proses pencarian multi-kata berdasarkan session state
        current_multi = st.session_state.get('multi_search_query', '')
        if current_multi.strip():
            if ',' in current_multi:
                # OR search
                words = [''.join(c for c in w.strip().lower() if c.isalnum()) for w in current_multi.split(',')]
                found_docs = set()
                found_words = []
                for word in words:
                    if word in search_index:
                        found_docs.update(search_index[word])
                        found_words.append(word)
                
                if found_docs and search_doc_list:
                    st.success(f"✅ Ditemukan di {len(found_docs)} dokumen (OR search): {', '.join(found_words)}")
                    for doc_id in sorted(found_docs):
                        if doc_id <= len(search_doc_list):
                            st.write(f"**📄 Dokumen {doc_id}:** {search_doc_list[doc_id - 1]}")
                else:
                    st.warning("⚠️ Tidak ada dokumen yang mengandung kata-kata tersebut.")
            else:
                # AND search
                words = [''.join(c for c in w.strip().lower() if c.isalnum()) for w in current_multi.split()]
                if all(w in search_index for w in words):
                    found_docs = set(search_index[words[0]])
                    for word in words[1:]:
                        found_docs &= set(search_index[word])
                    
                    if found_docs and search_doc_list:
                        st.success(f"✅ Ditemukan di {len(found_docs)} dokumen yang mengandung semua kata (AND search)")
                        for doc_id in sorted(found_docs):
                            if doc_id <= len(search_doc_list):
                                st.write(f"**📄 Dokumen {doc_id}:** {search_doc_list[doc_id - 1]}")
                    else:
                        st.warning("⚠️ Tidak ada dokumen yang mengandung semua kata tersebut.")
                else:
                    missing_words = [w for w in words if w not in search_index]
                    st.warning(f"⚠️ Kata berikut tidak ditemukan: {', '.join(missing_words)}")
        
        # Statistik
        if search_index and search_doc_list:
            st.markdown('<h4 class="subheader">📊 Statistik</h4>', unsafe_allow_html=True)
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Kata Unik", len(search_index))
            with col_stat2:
                st.metric("Total Dokumen", len(search_doc_list))
            with col_stat3:
                total_words = sum(sum(freq.values()) for freq in search_word_freq.values())
                st.metric("Total Kata (dengan duplikasi)", total_words)
            
            # Visualisasi - Top words
            try:
                index_df = st.session_state.get('index_df')
                if index_df is not None:
                    st.markdown('<h4 class="subheader">📈 Grafik Kata Terpopuler</h4>', unsafe_allow_html=True)
                    top_words = index_df.nlargest(15, 'Total Frekuensi')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_words.plot(x='Kata', y='Total Frekuensi', kind='barh', ax=ax, color='coral')
                    ax.set_xlabel('Total Frekuensi', fontsize=12)
                    ax.set_ylabel('Kata', fontsize=12)
                    ax.set_title('Top 15 Kata dengan Frekuensi Tertinggi', fontsize=14, pad=15)
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.info(f"Visualisasi tidak tersedia: {str(e)}")
            
            # Export hasil
            index_df = st.session_state.get('index_df')
            if index_df is not None:
                st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.markdown(create_download_link(index_df, "inverted_index.csv", "csv"), unsafe_allow_html=True)
                with col_exp2:
                    st.markdown(create_download_link(index_df, "inverted_index.xlsx", "excel"), unsafe_allow_html=True)
                
                # Tampilkan juga dalam format JSON
                with st.expander("📄 Tampilkan sebagai JSON"):
                    st.json(search_index)

# ===========================================
# PROGRAM 3: MODEL LDA
# ===========================================
elif program == "🎯 Model LDA":
    st.markdown('<h2 class="subheader">🎯 Model LDA (Latent Dirichlet Allocation)</h2>', unsafe_allow_html=True)
    st.write("Program ini melatih model LDA untuk menemukan topik-topik tersembunyi dalam kumpulan dokumen.")
    
    # Input parameter
    col1, col2 = st.columns(2)
    
    with col1:
        bahasa = st.selectbox(
            'Pilih Bahasa:',
            ['Indonesia', 'Inggris']
        )
    
    with col2:
        num_topics = st.slider(
            'Pilih Jumlah Topik:',
            min_value=2,
            max_value=10,
            value=3
        )
    
    # Upload file atau input manual
    input_method = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload File (CSV/TXT)"], horizontal=True)
    
    doc_list = []
    input_source = ""
    
    if input_method == "📁 Upload File (CSV/TXT)":
        uploaded_file = st.file_uploader("Pilih file", type=['csv', 'txt'])
        if uploaded_file is not None:
            doc_list, input_source = load_uploaded_file(uploaded_file)
            if doc_list is None:
                st.error(f"❌ {input_source}")
            else:
                st.success(f"✅ {input_source} berhasil dimuat! Ditemukan {len(doc_list)} dokumen.")
    else:
        # Default dokumen berdasarkan bahasa
        if bahasa == 'Indonesia':
            default_docs_lda = "Saya suka belajar bahasa pemrograman Python.\nData science adalah bidang yang menarik.\nMachine learning membantu dalam analisis data.\nPython digunakan untuk pengembangan aplikasi.\nAnalisis data membutuhkan keterampilan statistika.\nDeep learning adalah cabang dari machine learning.\nNatural language processing menggunakan Python."
        else:
            default_docs_lda = "I love learning Python programming language.\nData science is an interesting field.\nMachine learning helps in data analysis.\nPython is used for application development.\nData analysis requires statistical skills.\nDeep learning is a branch of machine learning.\nNatural language processing uses Python."
        
        documents = st.text_area(
            'Masukkan dokumen (satu per baris):',
            value=default_docs_lda,
            height=200,
            help="Masukkan minimal 5-6 dokumen untuk hasil yang lebih baik"
        )
        if documents.strip():
            doc_list = [doc.strip() for doc in documents.split('\n') if doc.strip()]
    
    if st.button('🚀 Latih Model LDA', type="primary"):
        if not doc_list or len(doc_list) < 3:
            st.warning("⚠️ Minimal diperlukan 3 dokumen untuk pelatihan LDA.")
        elif doc_list:
            with st.spinner('⏳ Sedang melatih model LDA...'):
                try:
                        # Tentukan bahasa untuk stopwords
                        lang_map = {
                            'Indonesia': 'indonesian',
                            'Inggris': 'english'
                        }
                        lang_code = lang_map[bahasa]
                        
                        # Muat stopwords
                        try:
                            stop_words = set(stopwords.words(lang_code))
                        except:
                            # Fallback jika bahasa tidak tersedia
                            if bahasa == 'Indonesia':
                                stop_words = {'yang', 'di', 'ke', 'dan', 'atau', 'dari', 'pada', 'adalah', 'ini', 'itu'}
                            else:
                                stop_words = set(stopwords.words('english'))
                        
                        # Fungsi preprocessing
                        def preprocess(text):
                            """Preprocessing teks: tokenisasi, lowercase, hapus stopwords dan non-alfabet"""
                            try:
                                tokens = word_tokenize(text.lower())
                                # Filter: hanya alfabet, bukan stopwords, panjang > 2
                                tokens = [token for token in tokens 
                                         if token.isalpha() 
                                         and token not in stop_words 
                                         and len(token) > 2]
                                return tokens
                            except:
                                # Fallback tokenisasi sederhana
                                tokens = text.lower().split()
                                tokens = [token for token in tokens 
                                         if token.isalpha() 
                                         and token not in stop_words 
                                         and len(token) > 2]
                                return tokens
                        
                        # Preprocess semua dokumen
                        processed_texts = [preprocess(doc) for doc in doc_list]
                        
                        # Filter dokumen yang kosong setelah preprocessing
                        processed_texts = [doc for doc in processed_texts if doc]
                        
                        if not processed_texts:
                            st.error("❌ Setelah preprocessing, tidak ada dokumen yang valid. Silakan gunakan dokumen yang lebih panjang.")
                        elif len(processed_texts) < 2:
                            st.warning("⚠️ Setelah preprocessing, hanya tersisa sedikit dokumen. Silakan tambahkan lebih banyak dokumen.")
                        else:
                            # Buat dictionary dan corpus
                            dictionary = corpora.Dictionary(processed_texts)
                            corpus = [dictionary.doc2bow(text) for text in processed_texts]
                            
                            # Latih model LDA
                            lda_model = LdaModel(
                                corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                random_state=42,
                                passes=15,
                                alpha='auto',
                                per_word_topics=True
                            )
                            
                            # Tampilkan hasil
                            st.markdown('<h3 class="subheader">📊 Topik yang Ditemukan</h3>', unsafe_allow_html=True)
                            
                            topics = lda_model.print_topics(num_words=10)
                            topics_data = []
                            for i, topic in enumerate(topics):
                                # Parse topic string untuk mendapatkan kata-kata
                                topic_str = topic[1]
                                words = [w.split('*')[1].strip('"') for w in topic_str.split('+')]
                                topics_data.append({
                                    'Topik': f'Topik {i}',
                                    'Kata Kunci (Top 10)': ', '.join(words[:10])
                                })
                                st.markdown(f"**Topik {i+1}:**")
                                st.write(topic[1])
                                st.write("")
                            
                            topics_df = pd.DataFrame(topics_data)
                            
                            # Visualisasi Interaktif dengan pyLDAvis
                            if PYLDAVIS_AVAILABLE:
                                st.markdown('<h3 class="subheader">📈 Visualisasi Interaktif (pyLDAvis)</h3>', unsafe_allow_html=True)
                                try:
                                    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
                                    html_string = pyLDAvis.prepared_data_to_html(vis)
                                    # type: ignore[attr-defined] - streamlit.components exists
                                    import streamlit.components.v1 as components  # type: ignore
                                    components.html(html_string, width=1300, height=800, scrolling=True)
                                except Exception as e:
                                    st.info(f"Visualisasi pyLDAvis tidak tersedia: {str(e)}")
                            
                            # Word Cloud per topik
                            st.markdown('<h3 class="subheader">☁️ Word Cloud per Topik</h3>', unsafe_allow_html=True)
                            try:
                                num_cols = 2
                                cols_wc = st.columns(num_cols)
                                for idx, topic in enumerate(topics):
                                    col_idx = idx % num_cols
                                    with cols_wc[col_idx]:
                                        # Ambil kata-kata dari topik
                                        topic_str = topic[1]
                                        words_dict = {}
                                        for item in topic_str.split('+'):
                                            parts = item.split('*')
                                            if len(parts) == 2:
                                                score = float(parts[0].strip())
                                                word = parts[1].strip('"').strip()
                                                words_dict[word] = score
                                        
                                        # Buat word cloud
                                        if words_dict:
                                            text_for_wc = ' '.join([word for word, _ in sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:30]])
                                            wc_fig = create_wordcloud(text_for_wc, f"Topik {idx+1}")
                                            if wc_fig:
                                                st.pyplot(wc_fig)
                                                plt.close()
                            except Exception as e:
                                st.info(f"Word cloud tidak tersedia: {str(e)}")
                            
                            # Distribusi topik per dokumen
                            st.markdown('<h3 class="subheader">📊 Distribusi Topik per Dokumen</h3>', unsafe_allow_html=True)
                            try:
                                doc_topics = []
                                for doc_bow in corpus:
                                    topic_dist = lda_model.get_document_topics(doc_bow)
                                    topics_scores = {f'Topik {tid}': score for tid, score in topic_dist}
                                    doc_topics.append(topics_scores)
                                
                                doc_topics_df = pd.DataFrame(doc_topics)
                                doc_topics_df.index = [f'Dokumen {i+1}' for i in range(len(doc_list))]
                                
                                # Visualisasi
                                fig, ax = plt.subplots(figsize=(12, max(6, len(doc_list) * 0.3)))
                                doc_topics_df.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')
                                ax.set_xlabel('Distribusi Topik', fontsize=12)
                                ax.set_ylabel('Dokumen', fontsize=12)
                                ax.set_title('Distribusi Topik per Dokumen', fontsize=14, pad=15)
                                ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                                st.dataframe(doc_topics_df, use_container_width=True)
                            except Exception as e:
                                st.info(f"Visualisasi distribusi tidak tersedia: {str(e)}")
                            
                            # Tampilkan informasi tambahan
                            with st.expander("ℹ️ Informasi Model"):
                                st.write(f"**Jumlah Dokumen:** {len(doc_list)}")
                                st.write(f"**Jumlah Topik:** {num_topics}")
                                st.write(f"**Ukuran Vocabulary:** {len(dictionary)}")
                            
                            # Export hasil
                            st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
                            col_exp1, col_exp2 = st.columns(2)
                            with col_exp1:
                                st.markdown(create_download_link(topics_df, "lda_topics.csv", "csv"), unsafe_allow_html=True)
                            with col_exp2:
                                st.markdown(create_download_link(topics_df, "lda_topics.xlsx", "excel"), unsafe_allow_html=True)
                            
                            if 'doc_topics_df' in locals():
                                col_exp3, col_exp4 = st.columns(2)
                                with col_exp3:
                                    st.markdown(create_download_link(doc_topics_df, "lda_document_distribution.csv", "csv"), unsafe_allow_html=True)
                                with col_exp4:
                                    st.markdown(create_download_link(doc_topics_df, "lda_document_distribution.xlsx", "excel"), unsafe_allow_html=True)
                            
                            st.success(f"✅ Model LDA berhasil dilatih dengan {num_topics} topik!")
                
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
                    st.info("💡 Tips: Pastikan dokumen yang Anda masukkan cukup dan berisi teks yang bermakna.")
        else:
            st.error("❌ Silakan masukkan dokumen terlebih dahulu.")

# ===========================================
# PROGRAM 4: MODEL BERTOPIC
# ===========================================
elif program == "🤖 Model BERTopic":
    st.markdown('<h2 class="subheader">🤖 Model BERTopic</h2>', unsafe_allow_html=True)
    st.write("Program ini menggunakan BERTopic untuk menemukan topik-topik dalam dokumen dengan menggunakan embedding berbasis transformer.")
    
    if not BERTOPIC_AVAILABLE:
        st.error("❌ BERTopic tidak tersedia. Silakan install dengan: pip install bertopic sentence-transformers")
        st.stop()
    
    # Input parameter
    bahasa = st.selectbox(
        'Pilih Bahasa:',
        ['Indonesia', 'Inggris']
    )
    
    # Upload file atau input manual
    input_method = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload File (CSV/TXT)"], horizontal=True)
    
    doc_list = []
    input_source = ""
    
    if input_method == "📁 Upload File (CSV/TXT)":
        uploaded_file = st.file_uploader("Pilih file", type=['csv', 'txt'])
        if uploaded_file is not None:
            doc_list, input_source = load_uploaded_file(uploaded_file)
            if doc_list is None:
                st.error(f"❌ {input_source}")
            else:
                st.success(f"✅ {input_source} berhasil dimuat! Ditemukan {len(doc_list)} dokumen.")
    else:
        # Default dokumen berdasarkan bahasa
        if bahasa == 'Indonesia':
            default_docs_bt = "Saya suka belajar bahasa pemrograman Python untuk analisis data.\nData science adalah bidang yang sangat menarik dan banyak peluangnya.\nMachine learning membantu dalam analisis data dan prediksi masa depan.\nPython digunakan untuk pengembangan aplikasi web dan mobile.\nAnalisis data membutuhkan keterampilan statistika dan matematika yang kuat.\nDeep learning adalah cabang dari machine learning yang sangat powerful.\nNatural language processing menggunakan Python untuk memahami teks.\nNeural network adalah fondasi dari deep learning dan artificial intelligence.\nPython memiliki banyak library untuk data science seperti pandas dan numpy.\nVisualisasi data penting untuk memahami informasi dengan lebih baik.\nBig data memerlukan tools khusus untuk processing yang efisien.\nCloud computing memungkinkan kita menyimpan data dalam jumlah besar."
        else:
            default_docs_bt = "I love learning Python programming language for data analysis.\nData science is a very interesting field with many opportunities.\nMachine learning helps in data analysis and future predictions.\nPython is used for web and mobile application development.\nData analysis requires strong statistical and mathematical skills.\nDeep learning is a very powerful branch of machine learning.\nNatural language processing uses Python to understand text.\nNeural networks are the foundation of deep learning and AI.\nPython has many libraries for data science like pandas and numpy.\nData visualization is important to understand information better.\nBig data requires special tools for efficient processing.\nCloud computing allows us to store large amounts of data."
        
        documents = st.text_area(
            'Masukkan dokumen (satu per baris):',
            value=default_docs_bt,
            height=200,
            help="Masukkan minimal 5-6 dokumen untuk hasil yang lebih baik"
        )
        if documents.strip():
            doc_list = [doc.strip() for doc in documents.split('\n') if doc.strip()]
    
    if st.button('🚀 Latih Model BERTopic', type="primary"):
        if not doc_list or len(doc_list) < 3:
            st.error(f"❌ Minimal diperlukan 3 dokumen. Anda memasukkan {len(doc_list) if doc_list else 0} dokumen.")
        elif doc_list:
            with st.spinner('⏳ Sedang melatih model BERTopic... (Ini mungkin membutuhkan waktu beberapa menit untuk pertama kali karena perlu mengunduh model)'):
                try:
                    # Validasi jumlah dokumen
                    if len(doc_list) < 5:
                        st.warning(f"⚠️ Anda memasukkan {len(doc_list)} dokumen. Untuk hasil terbaik, disarankan minimal 10 dokumen. Aplikasi akan tetap mencoba memproses dengan {len(doc_list)} dokumen yang ada.")
                    
                    if len(doc_list) >= 3:
                        # Hitung min_topic_size berdasarkan jumlah dokumen
                        # Untuk dataset kecil, gunakan min_topic_size yang lebih kecil
                        min_topic_size = max(2, len(doc_list) // 4)  # Minimal 2, maksimal 25% dari dokumen
                        
                        # Tentukan model embedding berdasarkan bahasa
                        if bahasa == 'Inggris':
                            embedding_model = 'all-MiniLM-L6-v2'
                            topic_model = BERTopic(
                                embedding_model=embedding_model,
                                min_topic_size=min_topic_size,
                                verbose=False,
                                calculate_probabilities=True
                            )
                        else:  # Indonesia
                            embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
                            topic_model = BERTopic(
                                language='multilingual',
                                embedding_model=embedding_model,
                                min_topic_size=min_topic_size,
                                verbose=False,
                                calculate_probabilities=True
                            )
                        
                        # Latih model dengan error handling tambahan
                        try:
                            topics, probs = topic_model.fit_transform(doc_list)
                        except ValueError as ve:
                            # Jika masih error, coba dengan min_topic_size yang lebih kecil
                            if "k must be less than" in str(ve) or "min_cluster_size" in str(ve).lower():
                                min_topic_size = 2
                                if bahasa == 'Inggris':
                                    topic_model = BERTopic(
                                        embedding_model='all-MiniLM-L6-v2',
                                        min_topic_size=min_topic_size,
                                        verbose=False,
                                        calculate_probabilities=False
                                    )
                                else:
                                    topic_model = BERTopic(
                                        language='multilingual',
                                        embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
                                        min_topic_size=min_topic_size,
                                        verbose=False,
                                        calculate_probabilities=False
                                    )
                                topics, probs = topic_model.fit_transform(doc_list)
                            else:
                                raise
                        
                        # Tampilkan hasil
                        st.markdown('<h3 class="subheader">📊 Info Topik</h3>', unsafe_allow_html=True)
                        
                        # Ambil info topik
                        topic_info = topic_model.get_topic_info()
                        st.dataframe(topic_info, use_container_width=True)
                        
                        # Tampilkan kata kunci untuk setiap topik
                        st.markdown('<h3 class="subheader">🔑 Kata Kunci per Topik</h3>', unsafe_allow_html=True)
                        
                        # Ambil topik (exclude -1 yang merupakan outlier)
                        topic_ids = [tid for tid in topic_info['Topic'].tolist() if tid != -1]
                        
                        for topic_id in topic_ids:
                            words = topic_model.get_topic(topic_id)
                            if words:
                                word_list = [word for word, score in words[:10]]  # Ambil 10 kata teratas
                                st.write(f"**Topik {topic_id}:** {', '.join(word_list)}")
                        
                        # Visualisasi
                        st.markdown('<h3 class="subheader">📈 Visualisasi Jarak Antar Topik</h3>', unsafe_allow_html=True)
                        
                        # Validasi: visualize_topics memerlukan minimal 2 topik (selain outlier)
                        if len(topic_ids) >= 2:
                            try:
                                fig = topic_model.visualize_topics()
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as viz_error:
                                error_msg = str(viz_error)
                                if "zero-size array" in error_msg or "maximum which has no identity" in error_msg:
                                    st.warning("⚠️ Visualisasi jarak antar topik memerlukan minimal 2 topik yang valid dengan embedding yang cukup. "
                                             f"Saat ini ditemukan {len(topic_ids)} topik. Coba tambahkan lebih banyak dokumen untuk mendapatkan lebih banyak topik.")
                                else:
                                    st.info(f"Visualisasi tidak tersedia: {error_msg}")
                        else:
                            st.warning(f"⚠️ Visualisasi jarak antar topik memerlukan minimal 2 topik yang valid. "
                                     f"Saat ini ditemukan {len(topic_ids)} topik. Silakan tambahkan lebih banyak dokumen untuk mendapatkan lebih banyak topik.")
                        
                        # Visualisasi barchart topik
                        if len(topic_ids) > 0:
                            try:
                                st.markdown('<h3 class="subheader">📊 Visualisasi Topik (Bar Chart)</h3>', unsafe_allow_html=True)
                                fig_barchart = topic_model.visualize_barchart(top_n_topics=min(10, len(topic_ids)))
                                st.plotly_chart(fig_barchart, use_container_width=True)
                            except Exception as bar_error:
                                st.info(f"Visualisasi bar chart tidak tersedia: {str(bar_error)}")
                        else:
                            st.info("ℹ️ Tidak ada topik yang ditemukan untuk divisualisasikan. Coba tambahkan lebih banyak dokumen.")
                        
                        # Export hasil
                        st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            st.markdown(create_download_link(topic_info, "bertopic_info.csv", "csv"), unsafe_allow_html=True)
                        with col_exp2:
                            st.markdown(create_download_link(topic_info, "bertopic_info.xlsx", "excel"), unsafe_allow_html=True)
                        
                        # Export dokumen dengan topik
                        try:
                            doc_topics_bt = pd.DataFrame({
                                'Dokumen': doc_list,
                                'Topik': topics,
                                'Probabilitas': [probs[i] if probs is not None and i < len(probs) else None for i in range(len(doc_list))]
                            })
                            col_exp3, col_exp4 = st.columns(2)
                            with col_exp3:
                                st.markdown(create_download_link(doc_topics_bt, "bertopic_documents.csv", "csv"), unsafe_allow_html=True)
                            with col_exp4:
                                st.markdown(create_download_link(doc_topics_bt, "bertopic_documents.xlsx", "excel"), unsafe_allow_html=True)
                        except:
                            pass
                        
                        st.success(f"✅ Model BERTopic berhasil dilatih! Ditemukan {len(topic_ids)} topik dari {len(doc_list)} dokumen.")
                
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Terjadi kesalahan: {error_msg}")
                    
                    # Dapatkan jumlah dokumen jika tersedia
                    try:
                        doc_list = [doc.strip() for doc in documents.split('\n') if doc.strip()]
                        num_docs = len(doc_list)
                    except:
                        num_docs = 0
                    
                    # Pesan bantuan spesifik berdasarkan jenis error
                    if "k must be less than" in error_msg or "min_cluster_size" in error_msg.lower():
                        st.info("💡 **Tips:** Error ini biasanya terjadi ketika jumlah dokumen terlalu sedikit. Coba tambahkan lebih banyak dokumen (minimal 10 dokumen untuk hasil yang lebih baik).")
                    elif "embedding" in error_msg.lower() or "download" in error_msg.lower():
                        st.info("💡 **Tips:** Pastikan koneksi internet Anda aktif. Model embedding sedang diunduh untuk pertama kali (ini membutuhkan waktu beberapa menit).")
                    elif num_docs > 0 and num_docs < 10:
                        st.info(f"💡 **Tips:** Anda hanya memasukkan {num_docs} dokumen. Untuk BERTopic, disarankan menggunakan minimal 10-15 dokumen untuk hasil yang optimal.")
                    else:
                        st.info("💡 **Tips:** Pastikan dokumen yang Anda masukkan cukup panjang dan bermakna. Dokumen yang terlalu pendek atau tidak relevan dapat menyebabkan error.")
        else:
            st.error("❌ Silakan masukkan dokumen terlebih dahulu.")

# ===========================================
# PROGRAM 5: WORD VECTOR REPRESENTATIONS
# ===========================================
elif program == "🔤 Word Vector Representations":
    st.markdown('<h2 class="subheader">🔤 Word Vector Representations</h2>', unsafe_allow_html=True)
    st.write("Program ini melatih model Word Vector (Word2Vec, FastText) untuk menghasilkan representasi vektor dari kata-kata.")
    
    # Parameter
    col1, col2 = st.columns(2)
    with col1:
        model_options = ['Word2Vec']
        if FASTTEXT_AVAILABLE:
            model_options.append('FastText')
        model_type = st.selectbox(
            'Pilih Model:',
            model_options
        )
        if not FASTTEXT_AVAILABLE and model_type == 'FastText':
            st.warning("⚠️ FastText tidak tersedia. Menggunakan Word2Vec sebagai alternatif.")
    with col2:
        vector_size = st.slider(
            'Ukuran Vektor:',
            min_value=50,
            max_value=300,
            value=100,
            step=50
        )
    
    # Upload file atau input manual
    input_method = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload File (CSV/TXT)"], horizontal=True)
    
    doc_list = []
    
    if input_method == "📁 Upload File (CSV/TXT)":
        uploaded_file = st.file_uploader("Pilih file", type=['csv', 'txt'])
        if uploaded_file is not None:
            doc_list, input_source = load_uploaded_file(uploaded_file)
            if doc_list is None:
                st.error(f"❌ {input_source}")
            else:
                st.success(f"✅ {input_source} berhasil dimuat! Ditemukan {len(doc_list)} dokumen.")
    else:
        default_docs_wv = "Saya suka belajar machine learning.\nDeep learning adalah cabang dari machine learning.\nNatural language processing menggunakan neural network.\nPython adalah bahasa pemrograman yang populer.\nData science memerlukan keterampilan analisis.\nNeural network dapat memproses data kompleks.\nMachine learning membantu dalam prediksi.\nArtificial intelligence berkembang dengan pesat.\nDeep learning menggunakan banyak layer.\nNatural language processing memahami teks."
        documents = st.text_area(
            'Masukkan dokumen (satu per baris):',
            value=default_docs_wv,
            height=200,
            help="Masukkan setiap dokumen dalam baris terpisah"
        )
        if documents.strip():
            doc_list = [doc.strip() for doc in documents.split('\n') if doc.strip()]
    
    if st.button('🚀 Latih Model Word Vector', type="primary"):
        if not doc_list or len(doc_list) < 3:
            st.warning("⚠️ Minimal diperlukan 3 dokumen untuk pelatihan Word Vector.")
        elif doc_list:
            with st.spinner('⏳ Sedang melatih model Word Vector...'):
                try:
                    # Preprocessing
                    def preprocess_for_wv(texts):
                        """Preprocess teks untuk Word Vector training"""
                        processed = []
                        for text in texts:
                            tokens = word_tokenize(text.lower())
                            tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
                            if tokens:
                                processed.append(tokens)
                        return processed
                    
                    processed_texts = preprocess_for_wv(doc_list)
                    
                    if len(processed_texts) < 2:
                        st.error("❌ Setelah preprocessing, tidak ada cukup dokumen yang valid.")
                    else:
                        # Latih model
                        if model_type == 'Word2Vec':
                            model = Word2Vec(
                                sentences=processed_texts,
                                vector_size=vector_size,
                                window=5,
                                min_count=1,
                                workers=4,
                                sg=0  # 0 untuk CBOW, 1 untuk Skip-gram
                            )
                        else:  # FastText
                            if not FASTTEXT_AVAILABLE:
                                st.error("❌ FastText tidak tersedia. Silakan install dengan: pip install fasttext")
                                st.stop()
                            model = FastText(
                                sentences=processed_texts,
                                vector_size=vector_size,
                                window=5,
                                min_count=1,
                                workers=4,
                                sg=0
                            )
                        
                        # Simpan model dan info ke session_state
                        st.session_state['wv_model'] = model
                        st.session_state['wv_model_type'] = model_type
                        st.session_state['wv_vector_size'] = vector_size
                        st.session_state['wv_vocab_size'] = len(model.wv.key_to_index)
                        st.session_state['wv_doc_count'] = len(processed_texts)
                        st.session_state['wv_model_trained'] = True
                        
                        st.success(f"✅ Model {model_type} berhasil dilatih!")
                
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
                    st.info("💡 Tips: Pastikan dokumen yang Anda masukkan cukup dan berisi teks yang bermakna.")
    
    # Tampilkan informasi model dan fitur search jika model sudah dilatih
    if st.session_state.get('wv_model_trained', False):
        model = st.session_state.get('wv_model')
        model_type = st.session_state.get('wv_model_type', 'Word2Vec')
        vector_size = st.session_state.get('wv_vector_size', 100)
        vocab_size = st.session_state.get('wv_vocab_size', 0)
        doc_count = st.session_state.get('wv_doc_count', 0)
        
        # Tampilkan informasi model
        st.markdown('<h3 class="subheader">ℹ️ Informasi Model</h3>', unsafe_allow_html=True)
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Ukuran Vektor", vector_size)
        with col_info2:
            st.metric("Jumlah Kata", vocab_size)
        with col_info3:
            st.metric("Jumlah Dokumen", doc_count)
        
        # Fitur: Cari kata yang mirip
        st.markdown('<h3 class="subheader">🔍 Cari Kata yang Mirip</h3>', unsafe_allow_html=True)
        
        # Initialize session state untuk search word
        if 'wv_search_word' not in st.session_state:
            st.session_state['wv_search_word'] = ''
        
        search_word = st.text_input(
            "Masukkan kata untuk mencari kata yang mirip:", 
            value=st.session_state['wv_search_word'],
            placeholder="contoh: machine",
            key='wv_search_input'
        )
        
        # Update session state
        st.session_state['wv_search_word'] = search_word
        
        if search_word.strip():
            search_word_lower = search_word.lower().strip()
            if search_word_lower in model.wv.key_to_index:
                try:
                    similar_words = model.wv.most_similar(search_word_lower, topn=10)
                    st.success(f"✅ Kata yang mirip dengan '{search_word}':")
                    similar_df = pd.DataFrame(similar_words, columns=['Kata', 'Similarity Score'])
                    st.dataframe(similar_df, use_container_width=True)
                    
                    # Visualisasi similarity
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        words = [w[0] for w in similar_words]
                        scores = [w[1] for w in similar_words]
                        ax.barh(words, scores, color='steelblue')
                        ax.set_xlabel('Similarity Score', fontsize=12)
                        ax.set_ylabel('Kata', fontsize=12)
                        ax.set_title(f'Top 10 Kata yang Mirip dengan "{search_word}"', fontsize=14, pad=15)
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    except:
                        pass
                except Exception as e:
                    st.warning(f"⚠️ Tidak dapat mencari kata yang mirip: {str(e)}")
            else:
                st.warning(f"⚠️ Kata '{search_word}' tidak ditemukan dalam vocabulary model.")
        
        # Fitur: Hitung similarity antara dua kata
        st.markdown('<h4 class="subheader">📊 Similarity Antara Dua Kata</h4>', unsafe_allow_html=True)
        
        # Initialize session state untuk similarity words
        if 'wv_word1' not in st.session_state:
            st.session_state['wv_word1'] = ''
        if 'wv_word2' not in st.session_state:
            st.session_state['wv_word2'] = ''
        
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            word1 = st.text_input(
                "Kata 1:", 
                value=st.session_state['wv_word1'],
                placeholder="contoh: machine",
                key='wv_word1_input'
            )
            st.session_state['wv_word1'] = word1
        with col_sim2:
            word2 = st.text_input(
                "Kata 2:", 
                value=st.session_state['wv_word2'],
                placeholder="contoh: learning",
                key='wv_word2_input'
            )
            st.session_state['wv_word2'] = word2
        
        if word1.strip() and word2.strip():
            word1_lower = word1.lower().strip()
            word2_lower = word2.lower().strip()
            if word1_lower in model.wv.key_to_index and word2_lower in model.wv.key_to_index:
                try:
                    similarity = model.wv.similarity(word1_lower, word2_lower)
                    st.metric(f"Similarity antara '{word1}' dan '{word2}'", f"{similarity:.4f}")
                    
                    # Interpretasi
                    if similarity > 0.7:
                        st.success("✅ Kedua kata sangat mirip!")
                    elif similarity > 0.4:
                        st.info("ℹ️ Kedua kata cukup mirip.")
                    else:
                        st.warning("⚠️ Kedua kata kurang mirip.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                missing = []
                if word1_lower not in model.wv.key_to_index:
                    missing.append(word1)
                if word2_lower not in model.wv.key_to_index:
                    missing.append(word2)
                st.warning(f"⚠️ Kata berikut tidak ditemukan: {', '.join(missing)}")
        
        # Tampilkan vocabulary
        with st.expander("📝 Lihat Vocabulary (Top 50 Kata)"):
            vocab_list = list(model.wv.key_to_index.keys())[:50]
            vocab_df = pd.DataFrame({'Kata': vocab_list})
            st.dataframe(vocab_df, use_container_width=True)
            st.write(f"Total vocabulary: {len(model.wv.key_to_index)} kata")
        
        # Export model info
        st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
        vocab_all = pd.DataFrame({
            'Kata': list(model.wv.key_to_index.keys()),
            'Index': list(model.wv.key_to_index.values())
        })
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown(create_download_link(vocab_all, "word_vector_vocab.csv", "csv"), unsafe_allow_html=True)
        with col_exp2:
            st.markdown(create_download_link(vocab_all, "word_vector_vocab.xlsx", "excel"), unsafe_allow_html=True)

# ===========================================
# PROGRAM 6: RECURRENT NEURAL NETWORK (RNN)
# ===========================================
elif program == "🧠 Recurrent Neural Network (RNN)":
    st.markdown('<h2 class="subheader">🧠 Recurrent Neural Network (RNN)</h2>', unsafe_allow_html=True)
    st.write("Program ini melatih model RNN untuk klasifikasi teks atau generasi teks menggunakan LSTM/GRU.")
    
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow tidak tersedia. Silakan install dengan: pip install tensorflow")
    else:
        # Pilih task
        task_type = st.selectbox(
            'Pilih Tipe Task:',
            ['Text Classification', 'Text Generation']
        )
        
        if task_type == 'Text Classification':
            st.info("ℹ️ **Text Classification**: Klasifikasi dokumen ke dalam kategori tertentu.")
            
            # Upload file atau input manual
            input_method = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload File (CSV/TXT)"], horizontal=True)
            
            doc_list = []
            labels = []
            
            if input_method == "📁 Upload File (CSV/TXT)":
                uploaded_file = st.file_uploader("Pilih file CSV dengan kolom 'text' dan 'label'", type=['csv'])
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns and 'label' in df.columns:
                            doc_list = df['text'].dropna().astype(str).tolist()
                            labels = df['label'].dropna().astype(str).tolist()
                            st.success(f"✅ File berhasil dimuat! Ditemukan {len(doc_list)} dokumen.")
                        else:
                            st.error("❌ File CSV harus memiliki kolom 'text' dan 'label'")
                    except Exception as e:
                        st.error(f"❌ Error membaca file: {str(e)}")
            else:
                st.write("**Contoh Format:** Setiap baris: `teks|label`")
                default_data = "Saya suka belajar machine learning|positif\nDeep learning sangat menarik|positif\nSaya tidak suka matematika|negatif\nPython adalah bahasa yang mudah|positif\nPemrograman itu sulit|negatif"
                data_input = st.text_area(
                    'Masukkan data (format: teks|label, satu per baris):',
                    value=default_data,
                    height=200
                )
                if data_input.strip():
                    lines = [line.strip() for line in data_input.split('\n') if line.strip()]
                    for line in lines:
                        if '|' in line:
                            parts = line.split('|', 1)
                            if len(parts) == 2:
                                doc_list.append(parts[0].strip())
                                labels.append(parts[1].strip())
            
            if st.button('🚀 Latih Model RNN', type="primary"):
                if len(doc_list) < 3 or len(doc_list) != len(labels):
                    st.error("❌ Minimal diperlukan 3 dokumen dengan label yang sesuai.")
                elif doc_list:
                    with st.spinner('⏳ Sedang melatih model RNN...'):
                        try:
                            # Preprocessing
                            tokenizer = Tokenizer()
                            tokenizer.fit_on_texts(doc_list)
                            sequences = tokenizer.texts_to_sequences(doc_list)
                            
                            # Padding
                            max_len = max(len(seq) for seq in sequences) if sequences else 100
                            max_len = min(max_len, 200)  # Limit max length
                            X = pad_sequences(sequences, maxlen=max_len, padding='post')
                            
                            # Encode labels
                            unique_labels = sorted(list(set(labels)))
                            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                            y = np.array([label_to_idx[label] for label in labels])
                            
                            # Split data
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Build model
                            vocab_size = len(tokenizer.word_index) + 1
                            embedding_dim = 50
                            
                            model = Sequential([
                                Embedding(vocab_size, embedding_dim, input_length=max_len),
                                LSTM(64, return_sequences=True),
                                LSTM(32),
                                Dense(len(unique_labels), activation='softmax')
                            ])
                            
                            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                            
                            # Train model
                            history = model.fit(
                                X_train, y_train,
                                epochs=10,
                                batch_size=32,
                                validation_data=(X_test, y_test),
                                verbose=0
                            )
                            
                            # Simpan model dan info ke session_state
                            st.session_state['rnn_model'] = model
                            st.session_state['rnn_tokenizer'] = tokenizer
                            st.session_state['rnn_max_len'] = max_len
                            st.session_state['rnn_unique_labels'] = unique_labels
                            st.session_state['rnn_history'] = history.history
                            st.session_state['rnn_model_trained'] = True
                            
                            st.success("✅ Model RNN berhasil dilatih!")
                        
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {str(e)}")
                            st.info("💡 Tips: Pastikan data memiliki format yang benar dan cukup untuk training.")
            
            # Tampilkan hasil training dan form prediksi jika model sudah dilatih
            if st.session_state.get('rnn_model_trained', False):
                model = st.session_state.get('rnn_model')
                tokenizer = st.session_state.get('rnn_tokenizer')
                max_len = st.session_state.get('rnn_max_len')
                unique_labels = st.session_state.get('rnn_unique_labels')
                history = st.session_state.get('rnn_history')
                
                # Tampilkan hasil
                st.markdown('<h3 class="subheader">📊 Hasil Pelatihan</h3>', unsafe_allow_html=True)
                
                # Metrics
                train_acc = history['accuracy'][-1]
                val_acc = history['val_accuracy'][-1]
                
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("Training Accuracy", f"{train_acc:.4f}")
                with col_met2:
                    st.metric("Validation Accuracy", f"{val_acc:.4f}")
                
                # Visualisasi training history
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history['accuracy'], label='Training')
                    ax1.plot(history['val_accuracy'], label='Validation')
                    ax1.set_title('Model Accuracy', fontsize=12)
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend()
                    
                    ax2.plot(history['loss'], label='Training')
                    ax2.plot(history['val_loss'], label='Validation')
                    ax2.set_title('Model Loss', fontsize=12)
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except:
                    pass
                
                # Test prediction
                st.markdown('<h3 class="subheader">🔮 Prediksi Teks Baru</h3>', unsafe_allow_html=True)
                
                # Initialize session state untuk test text
                if 'rnn_test_text' not in st.session_state:
                    st.session_state['rnn_test_text'] = ''
                
                test_text = st.text_input(
                    "Masukkan teks untuk diprediksi:", 
                    value=st.session_state['rnn_test_text'],
                    placeholder="contoh: Saya suka belajar AI",
                    key='rnn_test_input'
                )
                
                # Update session state
                st.session_state['rnn_test_text'] = test_text
                
                if test_text.strip():
                    test_seq = tokenizer.texts_to_sequences([test_text])
                    test_padded = pad_sequences(test_seq, maxlen=max_len, padding='post')
                    prediction = model.predict(test_padded, verbose=0)
                    predicted_idx = np.argmax(prediction[0])
                    predicted_label = unique_labels[predicted_idx]
                    confidence = prediction[0][predicted_idx]
                    
                    st.success(f"✅ Prediksi: **{predicted_label}** (Confidence: {confidence:.4f})")
                    
                    # Tampilkan semua probabilitas
                    prob_df = pd.DataFrame({
                        'Label': unique_labels,
                        'Probabilitas': prediction[0]
                    }).sort_values('Probabilitas', ascending=False)
                    st.dataframe(prob_df, use_container_width=True)
        
        else:  # Text Generation
            st.info("ℹ️ **Text Generation**: Generate teks baru berdasarkan teks yang diberikan.")
            
            # Input
            seed_text = st.text_area(
                'Masukkan seed text (teks awal):',
                value="Machine learning adalah",
                height=100,
                help="Teks ini akan digunakan sebagai awal untuk generate teks"
            )
            
            max_length = st.slider('Panjang teks yang di-generate:', min_value=10, max_value=100, value=50)
            
            if st.button('🚀 Generate Teks', type="primary"):
                if seed_text.strip():
                    with st.spinner('⏳ Sedang generate teks...'):
                        try:
                            # Simple text generation menggunakan model pre-trained
                            if TRANSFORMERS_AVAILABLE:
                                # Gunakan GPT-2 untuk text generation
                                generator = pipeline("text-generation", model="gpt2", device=-1)  # device=-1 untuk CPU
                                generated = generator(seed_text, max_length=max_length, num_return_sequences=1, temperature=0.7)
                                
                                st.markdown('<h3 class="subheader">📝 Teks yang Di-generate</h3>', unsafe_allow_html=True)
                                generated_text = generated[0]['generated_text']
                                st.write(generated_text)
                                
                                # Highlight seed text
                                highlighted = generated_text.replace(
                                    seed_text,
                                    f"**<mark style='background-color: yellow'>{seed_text}</mark>**"
                                )
                                st.markdown(f"**Full Text:** {highlighted}", unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ Transformers library tidak tersedia. Install dengan: pip install transformers torch")
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {str(e)}")
                            st.info("💡 Tips: Model akan diunduh untuk pertama kali (membutuhkan koneksi internet).")

# ===========================================
# PROGRAM 7: SEQUENCE TO SEQUENCE (SEQ2SEQ)
# ===========================================
elif program == "🔄 Sequence to Sequence (Seq2Seq)":
    st.markdown('<h2 class="subheader">🔄 Sequence to Sequence (Seq2Seq)</h2>', unsafe_allow_html=True)
    st.write("Program ini menggunakan model Seq2Seq untuk translation, summarization, atau task sequence-to-sequence lainnya.")
    
    if not TRANSFORMERS_AVAILABLE:
        st.error("❌ Transformers library tidak tersedia. Silakan install dengan: pip install transformers torch")
    else:
        # Pilih task
        task_type = st.selectbox(
            'Pilih Tipe Task:',
            ['Translation (Terjemahan)', 'Summarization (Ringkasan)', 'Question Answering']
        )
        
        if task_type == 'Translation (Terjemahan)':
            st.info("ℹ️ **Translation**: Menerjemahkan teks dari satu bahasa ke bahasa lain.")
            
            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                source_lang = st.selectbox('Bahasa Sumber:', ['English', 'Indonesian', 'French', 'German', 'Spanish'])
            with col_lang2:
                target_lang = st.selectbox('Bahasa Target:', ['Indonesian', 'English', 'French', 'German', 'Spanish'])
            
            # Model mapping
            model_map = {
                ('English', 'Indonesian'): 'Helsinki-NLP/opus-mt-en-id',
                ('Indonesian', 'English'): 'Helsinki-NLP/opus-mt-id-en',
                ('English', 'French'): 'Helsinki-NLP/opus-mt-en-fr',
                ('French', 'English'): 'Helsinki-NLP/opus-mt-fr-en',
            }
            
            model_name = model_map.get((source_lang, target_lang))
            if not model_name:
                model_name = 'Helsinki-NLP/opus-mt-en-id'  # Default
                st.info(f"ℹ️ Menggunakan model default untuk {source_lang} → {target_lang}")
            
            text_to_translate = st.text_area(
                'Masukkan teks yang ingin diterjemahkan:',
                value="Hello, how are you?" if source_lang == 'English' else "Halo, apa kabar?",
                height=150
            )
            
            if st.button('🌐 Terjemahkan', type="primary"):
                if text_to_translate.strip():
                    with st.spinner('⏳ Sedang menerjemahkan... (Model akan diunduh untuk pertama kali)'):
                        try:
                            translator = pipeline("translation", model=model_name, device=-1)
                            result = translator(text_to_translate)
                            translated_text = result[0]['translation_text']
                            
                            st.markdown('<h3 class="subheader">🌐 Hasil Terjemahan</h3>', unsafe_allow_html=True)
                            col_trans1, col_trans2 = st.columns(2)
                            with col_trans1:
                                st.write(f"**{source_lang}:**")
                                st.write(text_to_translate)
                            with col_trans2:
                                st.write(f"**{target_lang}:**")
                                st.success(translated_text)
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {str(e)}")
                            st.info("💡 Tips: Pastikan koneksi internet aktif. Model akan diunduh untuk pertama kali.")
        
        elif task_type == 'Summarization (Ringkasan)':
            st.info("ℹ️ **Summarization**: Membuat ringkasan dari teks yang panjang.")
            
            text_to_summarize = st.text_area(
                'Masukkan teks yang ingin diringkas:',
                value="Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Instead of being explicitly programmed to perform a task, machine learning systems learn from data. Deep learning, a subset of machine learning, uses neural networks with multiple layers to analyze various factors of data. Natural language processing is another important field that combines machine learning with linguistics to help computers understand human language.",
                height=200,
                help="Masukkan teks yang cukup panjang untuk diringkas"
            )
            
            max_length = st.slider('Panjang maksimal ringkasan:', min_value=30, max_value=150, value=50)
            min_length = st.slider('Panjang minimal ringkasan:', min_value=10, max_value=50, value=20)
            
            if st.button('📝 Buat Ringkasan', type="primary"):
                if text_to_summarize.strip():
                    with st.spinner('⏳ Sedang membuat ringkasan... (Model akan diunduh untuk pertama kali)'):
                        try:
                            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
                            result = summarizer(text_to_summarize, max_length=max_length, min_length=min_length, do_sample=False)
                            summary = result[0]['summary_text']
                            
                            st.markdown('<h3 class="subheader">📝 Ringkasan</h3>', unsafe_allow_html=True)
                            st.success(summary)
                            
                            # Perbandingan panjang
                            col_comp1, col_comp2 = st.columns(2)
                            with col_comp1:
                                st.metric("Panjang Teks Asli", f"{len(text_to_summarize.split())} kata")
                            with col_comp2:
                                st.metric("Panjang Ringkasan", f"{len(summary.split())} kata")
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {str(e)}")
                            st.info("💡 Tips: Pastikan koneksi internet aktif. Model akan diunduh untuk pertama kali.")
        
        else:  # Question Answering
            st.info("ℹ️ **Question Answering**: Menjawab pertanyaan berdasarkan konteks yang diberikan.")
            
            context = st.text_area(
                'Masukkan konteks (teks yang akan dijadikan referensi):',
                value="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                height=150
            )
            
            question = st.text_input(
                'Masukkan pertanyaan:',
                value="What is machine learning?",
                placeholder="contoh: What is machine learning?"
            )
            
            if st.button('❓ Jawab Pertanyaan', type="primary"):
                if context.strip() and question.strip():
                    with st.spinner('⏳ Sedang mencari jawaban... (Model akan diunduh untuk pertama kali)'):
                        try:
                            qa_pipeline = pipeline("question-answering", device=-1)
                            result = qa_pipeline(question=question, context=context)
                            
                            st.markdown('<h3 class="subheader">❓ Jawaban</h3>', unsafe_allow_html=True)
                            st.success(f"**{result['answer']}**")
                            st.metric("Confidence Score", f"{result['score']:.4f}")
                            
                            # Highlight answer dalam context
                            start_idx = result['start']
                            end_idx = result['end']
                            highlighted_context = (
                                context[:start_idx] +
                                f"**<mark style='background-color: yellow'>{context[start_idx:end_idx]}</mark>**" +
                                context[end_idx:]
                            )
                            st.markdown(f"**Konteks:** {highlighted_context}", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {str(e)}")
                            st.info("💡 Tips: Pastikan koneksi internet aktif. Model akan diunduh untuk pertama kali.")

# ===========================================
# PROGRAM 8: TRANSFORMERS
# ===========================================
elif program == "⚡ Transformers":
    st.markdown('<h2 class="subheader">⚡ Transformers</h2>', unsafe_allow_html=True)
    st.write("Program ini menggunakan model Transformer untuk berbagai task NLP seperti sentiment analysis, named entity recognition, dan text classification.")
    
    if not TRANSFORMERS_AVAILABLE:
        st.error("❌ Transformers library tidak tersedia. Silakan install dengan: pip install transformers torch")
    else:
        # Pilih task
        task_type = st.selectbox(
            'Pilih Tipe Task:',
            ['Sentiment Analysis', 'Named Entity Recognition (NER)', 'Text Classification', 'Zero-Shot Classification']
        )
        
        # Upload file atau input manual
        input_method = st.radio("Pilih metode input:", ["📝 Input Manual", "📁 Upload File (CSV/TXT)"], horizontal=True)
        
        doc_list = []
        
        if input_method == "📁 Upload File (CSV/TXT)":
            uploaded_file = st.file_uploader("Pilih file", type=['csv', 'txt'])
            if uploaded_file is not None:
                doc_list, input_source = load_uploaded_file(uploaded_file)
                if doc_list is None:
                    st.error(f"❌ {input_source}")
                else:
                    st.success(f"✅ {input_source} berhasil dimuat! Ditemukan {len(doc_list)} dokumen.")
        else:
            if task_type == 'Sentiment Analysis':
                default_text = "I love this product! It's amazing."
            elif task_type == 'Named Entity Recognition (NER)':
                default_text = "Barack Obama was born in Hawaii. He worked at the White House."
            elif task_type == 'Text Classification':
                default_text = "This is a technology article about artificial intelligence."
            else:
                default_text = "This is a great movie with excellent acting."
            
            text_input = st.text_area(
                'Masukkan teks:',
                value=default_text,
                height=150
            )
            if text_input.strip():
                doc_list = [text_input.strip()]
        
        if st.button('🚀 Proses dengan Transformer', type="primary"):
            if not doc_list:
                st.error("❌ Silakan masukkan teks terlebih dahulu.")
            else:
                with st.spinner('⏳ Sedang memproses... (Model akan diunduh untuk pertama kali jika diperlukan)'):
                    try:
                        if task_type == 'Sentiment Analysis':
                            classifier = pipeline("sentiment-analysis", device=-1)
                            results = []
                            for doc in doc_list:
                                result = classifier(doc)
                                results.append({
                                    'Teks': doc,
                                    'Label': result[0]['label'],
                                    'Score': result[0]['score']
                                })
                            
                            results_df = pd.DataFrame(results)
                            st.markdown('<h3 class="subheader">📊 Hasil Sentiment Analysis</h3>', unsafe_allow_html=True)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Visualisasi
                            try:
                                label_counts = results_df['Label'].value_counts()
                                fig, ax = plt.subplots(figsize=(8, 6))
                                label_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
                                ax.set_xlabel('Sentiment', fontsize=12)
                                ax.set_ylabel('Jumlah', fontsize=12)
                                ax.set_title('Distribusi Sentiment', fontsize=14, pad=15)
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            except:
                                pass
                            
                            # Export
                            st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
                            col_exp1, col_exp2 = st.columns(2)
                            with col_exp1:
                                st.markdown(create_download_link(results_df, "sentiment_analysis.csv", "csv"), unsafe_allow_html=True)
                            with col_exp2:
                                st.markdown(create_download_link(results_df, "sentiment_analysis.xlsx", "excel"), unsafe_allow_html=True)
                        
                        elif task_type == 'Named Entity Recognition (NER)':
                            ner_pipeline = pipeline("ner", aggregation_strategy="simple", device=-1)
                            all_results = []
                            for doc in doc_list:
                                entities = ner_pipeline(doc)
                                for entity in entities:
                                    all_results.append({
                                        'Teks': doc[:50] + '...' if len(doc) > 50 else doc,
                                        'Entity': entity['word'],
                                        'Label': entity['entity_group'],
                                        'Score': entity['score']
                                    })
                            
                            if all_results:
                                ner_df = pd.DataFrame(all_results)
                                st.markdown('<h3 class="subheader">📊 Hasil Named Entity Recognition</h3>', unsafe_allow_html=True)
                                st.dataframe(ner_df, use_container_width=True)
                                
                                # Visualisasi
                                try:
                                    label_counts = ner_df['Label'].value_counts()
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    label_counts.plot(kind='barh', ax=ax, color='coral')
                                    ax.set_xlabel('Jumlah', fontsize=12)
                                    ax.set_ylabel('Entity Type', fontsize=12)
                                    ax.set_title('Distribusi Entity Types', fontsize=14, pad=15)
                                    ax.invert_yaxis()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                except:
                                    pass
                                
                                # Export
                                st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
                                col_exp1, col_exp2 = st.columns(2)
                                with col_exp1:
                                    st.markdown(create_download_link(ner_df, "ner_results.csv", "csv"), unsafe_allow_html=True)
                                with col_exp2:
                                    st.markdown(create_download_link(ner_df, "ner_results.xlsx", "excel"), unsafe_allow_html=True)
                            else:
                                st.info("ℹ️ Tidak ada named entity yang ditemukan dalam teks.")
                        
                        elif task_type == 'Text Classification':
                            # Zero-shot classification untuk text classification
                            classifier = pipeline("zero-shot-classification", device=-1)
                            candidate_labels = st.text_input(
                                "Masukkan kategori (pisahkan dengan koma):",
                                value="technology, sports, politics, entertainment, science"
                            )
                            
                            if candidate_labels.strip():
                                labels = [l.strip() for l in candidate_labels.split(',')]
                                results = []
                                for doc in doc_list:
                                    result = classifier(doc, labels)
                                    results.append({
                                        'Teks': doc,
                                        'Predicted Label': result['labels'][0],
                                        'Score': result['scores'][0]
                                    })
                                
                                results_df = pd.DataFrame(results)
                                st.markdown('<h3 class="subheader">📊 Hasil Text Classification</h3>', unsafe_allow_html=True)
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Tampilkan semua scores untuk teks pertama
                                if doc_list:
                                    first_result = classifier(doc_list[0], labels)
                                    scores_df = pd.DataFrame({
                                        'Label': first_result['labels'],
                                        'Score': first_result['scores']
                                    })
                                    st.markdown('<h4 class="subheader">📈 Probabilitas Semua Kategori</h4>', unsafe_allow_html=True)
                                    st.dataframe(scores_df, use_container_width=True)
                                
                                # Export
                                st.markdown('<h3 class="subheader">💾 Export Hasil</h3>', unsafe_allow_html=True)
                                col_exp1, col_exp2 = st.columns(2)
                                with col_exp1:
                                    st.markdown(create_download_link(results_df, "text_classification.csv", "csv"), unsafe_allow_html=True)
                                with col_exp2:
                                    st.markdown(create_download_link(results_df, "text_classification.xlsx", "excel"), unsafe_allow_html=True)
                        
                        else:  # Zero-Shot Classification
                            text_to_classify = doc_list[0] if doc_list else ""
                            candidate_labels = st.text_input(
                                "Masukkan kategori untuk klasifikasi (pisahkan dengan koma):",
                                value="positive, negative, neutral"
                            )
                            
                            if candidate_labels.strip() and text_to_classify:
                                labels = [l.strip() for l in candidate_labels.split(',')]
                                classifier = pipeline("zero-shot-classification", device=-1)
                                result = classifier(text_to_classify, labels)
                                
                                st.markdown('<h3 class="subheader">📊 Hasil Zero-Shot Classification</h3>', unsafe_allow_html=True)
                                
                                # Tampilkan hasil
                                st.write(f"**Teks:** {text_to_classify}")
                                st.success(f"**Prediksi:** {result['labels'][0]} (Score: {result['scores'][0]:.4f})")
                                
                                # Tampilkan semua scores
                                scores_df = pd.DataFrame({
                                    'Label': result['labels'],
                                    'Score': result['scores']
                                })
                                st.dataframe(scores_df, use_container_width=True)
                                
                                # Visualisasi
                                try:
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    scores_df.plot(x='Label', y='Score', kind='barh', ax=ax, color='steelblue')
                                    ax.set_xlabel('Score', fontsize=12)
                                    ax.set_ylabel('Label', fontsize=12)
                                    ax.set_title('Zero-Shot Classification Scores', fontsize=14, pad=15)
                                    ax.invert_yaxis()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                except:
                                    pass
                        
                        st.success("✅ Proses selesai!")
                    
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan: {str(e)}")
                        st.info("💡 Tips: Pastikan koneksi internet aktif. Model akan diunduh untuk pertama kali (membutuhkan waktu beberapa menit).")

# ===========================================
# HALAMAN BERANDA
# ===========================================
else:
    st.markdown('<h2 class="subheader">🏠 Selamat Datang di NLP - One for All</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📖 Tentang Aplikasi
    
    Aplikasi ini menyediakan berbagai tools NLP (Natural Language Processing) yang dapat Anda gunakan untuk menganalisis dan memproses teks. 
    Pilih program yang ingin Anda gunakan dari menu sidebar di sebelah kiri.
    
    ### 🔧 Program yang Tersedia
    
    1. **🔍 Feature Extraction (TF-IDF)**
       - Mengekstraksi fitur dari dokumen menggunakan metode TF-IDF
       - Menghasilkan matriks TF-IDF yang menunjukkan pentingnya setiap kata dalam setiap dokumen
       - Visualisasi: Heatmap, Word Cloud, Bar Chart
       - Upload file dan Export hasil
       
    2. **📑 Inverted Index**
       - Membuat struktur data inverted index sederhana
       - Memetakan setiap kata ke dokumen-dokumen yang mengandung kata tersebut
       - Fitur pencarian: Pencarian kata tunggal dan multi-kata (AND/OR)
       - Upload file dan Export hasil
       
    3. **🎯 Model LDA (Latent Dirichlet Allocation)**
       - Melatih model LDA untuk menemukan topik-topik tersembunyi dalam dokumen
       - Mendukung dataset Bahasa Indonesia dan Inggris
       - Visualisasi interaktif pyLDAvis
       - Upload file dan Export hasil
       
    4. **🤖 Model BERTopic**
       - Menggunakan BERTopic untuk analisis topik dengan embedding transformer
       - Mendukung dataset Bahasa Indonesia dan Inggris
       - Menyediakan visualisasi interaktif
       - Upload file dan Export hasil
       
    5. **🔤 Word Vector Representations**
       - Melatih model Word2Vec dan FastText untuk representasi vektor kata
       - Mencari kata yang mirip (similarity)
       - Menghitung similarity antara dua kata
       - Upload file dan Export hasil
       
    6. **🧠 Recurrent Neural Network (RNN)**
       - Text Classification menggunakan LSTM
       - Text Generation menggunakan model pre-trained
       - Visualisasi training history
       - Prediksi teks baru
       
    7. **🔄 Sequence to Sequence (Seq2Seq)**
       - Translation (Terjemahan) antar bahasa
       - Summarization (Ringkasan teks)
       - Question Answering (Menjawab pertanyaan)
       - Menggunakan model transformer pre-trained
       
    8. **⚡ Transformers**
       - Sentiment Analysis
       - Named Entity Recognition (NER)
       - Text Classification
       - Zero-Shot Classification
       - Menggunakan model transformer modern
    
    ### 🚀 Cara Menggunakan
    
    1. Pilih program yang ingin Anda gunakan dari menu sidebar
    2. Ikuti instruksi di setiap halaman program
    3. Masukkan data dokumen Anda
    4. Klik tombol untuk memproses
    5. Lihat hasilnya!
    
    ---
    
    **💡 Tips:** Untuk hasil yang lebih baik, gunakan dokumen yang cukup panjang dan bermakna.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p>NLP - One for All | Dibuat dengan ❤️ menggunakan Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

