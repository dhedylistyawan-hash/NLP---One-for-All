# 📤 Panduan Upload Project ke GitHub

Panduan lengkap untuk mengupload project **NLP - One for All** ke GitHub.

## 📋 Persiapan

### 1. Install Git
Jika belum install Git, download dari: https://git-scm.com/downloads

### 2. Buat Akun GitHub
Jika belum punya akun GitHub, buat di: https://github.com/signup

---

## 🚀 Langkah-Langkah Upload

### METODE 1: Menggunakan Command Line (Terminal/Git Bash)

#### Langkah 1: Buka Terminal/Git Bash
Buka terminal di folder project Anda:
```bash
cd C:\Users\pusbi\OneDrive\Desktop\NLP
```

#### Langkah 2: Inisialisasi Git Repository
```bash
# Inisialisasi git repository
git init
```

#### Langkah 3: Tambahkan File ke Staging Area
```bash
# Tambahkan semua file
git add .

# Atau tambahkan file spesifik
git add app.py
git add requirements.txt
git add README.md
git add LAPORAN.md
git add .gitignore
```

#### Langkah 4: Commit File
```bash
# Commit dengan pesan
git commit -m "Initial commit: NLP - One for All aplikasi dengan TF-IDF, Inverted Index, LDA, dan BERTopic"
```

#### Langkah 5: Buat Repository di GitHub
1. Login ke GitHub
2. Klik ikon **+** di pojok kanan atas
3. Pilih **New repository**
4. Isi:
   - **Repository name**: `nlp-one-for-all` (atau nama lain yang Anda suka)
   - **Description**: "Aplikasi web NLP dengan TF-IDF, Inverted Index, LDA, dan BERTopic"
   - Pilih **Public** atau **Private**
   - JANGAN centang "Initialize this repository with a README" (karena kita sudah punya)
5. Klik **Create repository**

#### Langkah 6: Hubungkan dengan Remote Repository
```bash
# Ganti YOUR_USERNAME dengan username GitHub Anda
# Ganti REPO_NAME dengan nama repository yang Anda buat
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Contoh:
# git remote add origin https://github.com/username/nlp-one-for-all.git
```

#### Langkah 7: Push ke GitHub
```bash
# Push ke branch main
git branch -M main
git push -u origin main
```

Jika diminta username dan password:
- **Username**: Username GitHub Anda
- **Password**: Personal Access Token (bukan password GitHub biasa)

**Cara membuat Personal Access Token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Beri nama token (misalnya: "GitHub Upload")
4. Pilih scope: `repo` (centang semua checkbox di bawah)
5. Generate token
6. **SALIN TOKEN** (hanya muncul sekali!)
7. Gunakan token sebagai password saat push

---

### METODE 2: Menggunakan GitHub Desktop (Lebih Mudah)

#### Langkah 1: Install GitHub Desktop
Download dari: https://desktop.github.com/

#### Langkah 2: Login ke GitHub Desktop
Login dengan akun GitHub Anda

#### Langkah 3: Add Repository
1. Klik **File** → **Add Local Repository**
2. Pilih folder project: `C:\Users\pusbi\OneDrive\Desktop\NLP`
3. Klik **Add**

#### Langkah 4: Publish Repository
1. Klik **Publish repository** di GitHub Desktop
2. Isi nama repository (misalnya: `nlp-one-for-all`)
3. Tambahkan description (opsional)
4. Pilih **Public** atau **Private**
5. Klik **Publish Repository**

---

### METODE 3: Upload Manual via GitHub Web

#### Langkah 1: Buat Repository Baru di GitHub
1. Login ke GitHub
2. Klik **+** → **New repository**
3. Isi nama dan klik **Create repository**

#### Langkah 2: Upload File
1. Setelah repository dibuat, klik **uploading an existing file**
2. Drag & drop semua file dari folder project:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `LAPORAN.md`
   - `.gitignore`
   - `pyrightconfig.json`
3. Scroll ke bawah, isi commit message
4. Klik **Commit changes**

---

## ✅ Checklist File yang Harus Diupload

Pastikan file berikut ada di repository:

- ✅ `app.py` - File utama aplikasi
- ✅ `requirements.txt` - Daftar dependencies
- ✅ `README.md` - Dokumentasi proyek
- ✅ `LAPORAN.md` - Laporan lengkap
- ✅ `.gitignore` - File untuk mengabaikan file tertentu
- ✅ `pyrightconfig.json` - Konfigurasi type checker

### File yang TIDAK Perlu Diupload (sudah di .gitignore):
- ❌ `__pycache__/` - Cache Python
- ❌ `.venv/` atau `venv/` - Virtual environment
- ❌ File editor (`.vscode/`, `.idea/`)
- ❌ File sistem (`.DS_Store`, `Thumbs.db`)

---

## 🔍 Verifikasi Upload

Setelah upload berhasil:

1. **Cek di GitHub**: Buka repository Anda di browser
2. **Pastikan semua file ada**: File utama harus terlihat
3. **Cek README**: README.md harus tampil di halaman utama repository

---

## 🔄 Update Project di GitHub

Jika Anda melakukan perubahan dan ingin mengupdate:

### Menggunakan Command Line:
```bash
git add .
git commit -m "Update: deskripsi perubahan Anda"
git push origin main
```

### Menggunakan GitHub Desktop:
1. Klik **Commit** di GitHub Desktop
2. Tulis commit message
3. Klik **Push origin**

---

## 📝 Tips

1. **Commit Message yang Baik:**
   - "Initial commit: NLP application"
   - "Add TF-IDF feature extraction"
   - "Fix search functionality in Inverted Index"
   - "Update documentation"

2. **Repository Name yang Baik:**
   - `nlp-one-for-all`
   - `nlp-web-app`
   - `streamlit-nlp-toolkit`
   - Hindari spasi, gunakan tanda hubung (-)

3. **Description Repository:**
   - "Aplikasi web NLP dengan Streamlit: TF-IDF, Inverted Index, LDA, dan BERTopic"

---

## 🆘 Troubleshooting

### Error: "fatal: remote origin already exists"
```bash
# Hapus remote yang ada dulu
git remote remove origin

# Tambahkan lagi
git remote add origin https://github.com/USERNAME/REPO_NAME.git
```

### Error: "Failed to push some refs"
```bash
# Pull dulu sebelum push
git pull origin main --allow-unrelated-histories

# Lalu push lagi
git push origin main
```

### Error: "Authentication failed"
- Pastikan menggunakan Personal Access Token, bukan password
- Token harus memiliki permission `repo`

---

## 📚 Referensi

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [GitHub Desktop Guide](https://docs.github.com/en/desktop)

---

**Selamat! Project Anda sekarang sudah di GitHub! 🎉**

Jika ada masalah, jangan ragu untuk bertanya.

