from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import time
from huggingface_hub import hf_hub_download

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Ganti dengan kunci rahasia yang kuat untuk flash messages

# --- ID Repositori Hugging Face ---
# ID repositori untuk model Word2Vec
W2V_REPO_ID = "SukmaPutra/word2vec_model"
W2V_FILENAME = "word2vec_model.bin"

# ID repositori untuk model IndoBART
ABSTRACTIVE_MODEL_ID = "SukmaPutra/abstractive_model_artifacts" 

# --- Variabel Global untuk Model ---
word2vec_model = None
indobart_tokenizer = None
indobart_model = None

# --- Fungsi untuk mengunduh NLTK resources ---
def download_nltk_resources():
    print("Memeriksa dan mengunduh sumber daya NLTK (punkt, stopwords)...")
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)
    print("Sumber daya NLTK siap.")

# --- Fungsi untuk mengambil teks dari URL ---
def get_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        article_text = ""
        possible_content_tags = [
            {'name': 'article'},
            {'name': 'main'},
            {'name': 'div', 'class_': re.compile(r'article|content|post|story|body|entry-content', re.IGNORECASE)},
            {'name': 'section', 'class_': re.compile(r'article|content|post|story|body|entry-content', re.IGNORECASE)},
        ]

        for selector in possible_content_tags:
            tag_name = selector['name']
            tag_attrs = {k:v for k,v in selector.items() if k!='name'}
            found = soup.find(tag_name, **tag_attrs)
            if found:
                article_text = found.get_text(separator=' ', strip=True)
                if len(article_text.split()) > 50:
                    break
                else:
                    article_text = ""
        
        if not article_text or len(article_text.split()) < 50:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        if not article_text or len(article_text.split()) < 50:
            if soup.body:
                article_text = soup.body.get_text(separator=' ', strip=True)
            else:
                print("Tidak dapat menemukan tag <body> di halaman.")
                return None

        article_text = re.sub(r'\s+', ' ', article_text).strip()
        sentences = sent_tokenize(article_text)
        filtered_sentences = [
            s for s in sentences 
            if len(s.split()) > 5 and not re.search(r'^\s*(gambar|foto|video|iklan|advertisement|copyright|terkait|baca juga|ikuti kami|subscribe|newsletter|komentar|bagikan)\s*[:.]?\s*$', s, re.IGNORECASE)
        ]
        article_text = " ".join(filtered_sentences)
        return article_text if len(article_text.strip()) > 0 else None

    except requests.exceptions.RequestException as e:
        print(f"Gagal mengambil artikel dari URL: {e}. Pastikan URL valid dan dapat diakses.")
        return None
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses URL: {e}")
        return None

# --- Fungsi validasi URL sederhana ---
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# --- Fungsi Preprocessing untuk TextRank (Extractive) ---
def preprocess_single_article_extractive(article_content):
    cleaned_sentences_tokenized = []
    original_relevant_sentences = []
    stop_words = set(stopwords.words('indonesian'))
    sentences_from_content = sent_tokenize(article_content)
    for sentence in sentences_from_content:
        cleaned_sent_for_analysis = re.sub(r'\[baca:\s*[^\]]*\]', '', sentence, flags=re.IGNORECASE)
        cleaned_sent_for_analysis = re.sub(r'advertisement', '', cleaned_sent_for_analysis, flags=re.IGNORECASE)
        words = [word.lower() for word in word_tokenize(cleaned_sent_for_analysis)]
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        if filtered_words:
            cleaned_sentences_tokenized.append(filtered_words)
            original_relevant_sentences.append(sentence)
    return cleaned_sentences_tokenized, original_relevant_sentences

# --- Fungsi untuk mendapatkan vektor kalimat (Extractive) ---
def get_sentence_vector(sentence_tokens, word2vec_model):
    if word2vec_model is None:
        return np.zeros(100) 
    vectors = []
    for word in sentence_tokens:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(word2vec_model.vector_size)

# --- Fungsi TextRank untuk Peringkasan (Extractive) ---
def textrank_summarize(article_content, word2vec_model, target_word_count):
    all_original_sentences_in_order = sent_tokenize(article_content)
    fallback_summary = " ".join(all_original_sentences_in_order[:min(len(all_original_sentences_in_order), 2)])
    cleaned_sentences_tokenized, original_relevant_sentences = preprocess_single_article_extractive(article_content)
    if not cleaned_sentences_tokenized or len(cleaned_sentences_tokenized) < 2:
        return fallback_summary, "Gagal (Teks Sangat Pendek atau Tidak Relevan)"
    sentence_vectors = []
    valid_original_indices = []
    for i, tokens in enumerate(cleaned_sentences_tokenized):
        vec = get_sentence_vector(tokens, word2vec_model)
        if vec is not None and not np.all(vec == 0) and vec.shape[0] == word2vec_model.vector_size: 
            sentence_vectors.append(vec)
            valid_original_indices.append(i)
    if len(sentence_vectors) < 2:
        return fallback_summary, "Gagal (Vektor Kalimat Kurang)"
    try:
        similarity_matrix = cosine_similarity(sentence_vectors)
    except ValueError as e:
        return fallback_summary, f"Gagal (Masalah Dimensi Vektor: {e})"
    graph = nx.from_numpy_array(similarity_matrix)
    scores = {}
    try:
        scores = nx.pagerank(graph, max_iter=1000, tol=1e-3)
    except nx.PowerIterationFailedConvergence:
        print("Peringatan: PageRank gagal konvergen untuk teks ini. Menggunakan fallback.")
        return fallback_summary, "Gagal (PageRank Konvergensi)"
    ranked_processed_indices = sorted(((scores[i], idx_in_valid) for idx_in_valid, i in enumerate(scores)), reverse=True)
    current_word_count = 0
    final_summary_pairs = []
    num_selected_sentences = 0
    for score, processed_idx_in_valid in ranked_processed_indices:
        original_idx_in_relevant = valid_original_indices[processed_idx_in_valid]
        original_sentence = original_relevant_sentences[original_idx_in_relevant]
        try:
            actual_original_idx = all_original_sentences_in_order.index(original_sentence)
            if original_sentence not in [s for _, s in final_summary_pairs]:
                final_summary_pairs.append((actual_original_idx, original_sentence))
                current_word_count += len(word_tokenize(original_sentence))
                num_selected_sentences += 1
        except ValueError:
            pass
        if current_word_count >= target_word_count or num_selected_sentences >= len(original_relevant_sentences) * 0.3 + 1:
            break
    if not final_summary_pairs and len(all_original_sentences_in_order) > 0:
        return fallback_summary, "Gagal (Ringkasan Kosong/Terlalu Pendek)"
    final_summary_sentences_ordered = [sent for idx, sent in sorted(final_summary_pairs)]
    return " ".join(final_summary_sentences_ordered), "Sukses"

# --- Fungsi Pra-pemrosesan untuk Abstractive ---
def preprocess_abstractive(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Fungsi Peringkasan Abstractive dengan IndoBART ---
def indobart_summarize(text, tokenizer, model, min_len=30, max_len=150):
    clean_text = preprocess_abstractive(text)
    if len(clean_text.split()) > 512: 
        print("Peringatan: Teks input sangat panjang. Model abstractive mungkin hanya memproses 512 token awal. Untuk hasil terbaik, coba gunakan teks yang lebih ringkas.")
    with torch.no_grad(): 
        input_ids = tokenizer.encode(clean_text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(
            input_ids,
            min_length=min_len,
            max_length=max_len,
            num_beams=4,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            use_cache=True,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, "Sukses"

# --- Pemuatan Model (dilakukan sekali saat aplikasi dimulai) ---
def load_models():
    global word2vec_model, indobart_tokenizer, indobart_model
    
    download_nltk_resources()

    print("Mengunduh dan memuat model Word2Vec (Extractive)...")
    try:
        local_path = hf_hub_download(repo_id=W2V_REPO_ID, filename=W2V_FILENAME, local_dir=".")
        word2vec_model = Word2Vec.load(local_path)
        print("Model Word2Vec berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model Word2Vec dari Hugging Face Hub: {e}.")
        print(f"Pastikan ID repositori '{W2V_REPO_ID}' dan nama file '{W2V_FILENAME}' sudah benar, dan repositori tersebut bersifat publik.")
        word2vec_model = None

    print(f"Mengunduh dan memuat model IndoBART ({ABSTRACTIVE_MODEL_ID})... Ini mungkin memerlukan waktu beberapa saat.")
    try:
        indobart_tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL_ID)
        indobart_model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL_ID)
        indobart_model.eval()
        print("Model IndoBART berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model IndoBART dari Hugging Face Hub: {e}.")
        print(f"Pastikan ID model '{ABSTRACTIVE_MODEL_ID}' benar dan dapat diakses. Coba periksa koneksi internet Anda.")
        indobart_tokenizer, indobart_model = None, None

# Panggil fungsi pemuatan model saat aplikasi Flask dimulai
with app.app_context():
    load_models()

# --- ROUTES APLIKASI FLASK ---
@app.route('/', methods=['GET', 'POST'])
def index():
    user_input_text = ""
    extracted_text_preview = ""
    summary_result = ""
    summary_status = ""
    summary_type_selected = "Extractive (TextRank)" # Default selection
    extractive_word_count = 50 # Default
    abstractive_min_length = 30 # Default
    abstractive_max_length = 150 # Default

    if request.method == 'POST':
        input_source = request.form.get('input_source_radio')
        summary_type_selected = request.form.get('summary_type_radio')
        
        # Get length options
        try:
            extractive_word_count = int(request.form.get('extractive_word_count', 50))
            abstractive_min_length = int(request.form.get('abstractive_min_length', 30))
            abstractive_max_length = int(request.form.get('abstractive_max_length', 150))
        except ValueError:
            flash("Panjang ringkasan harus berupa angka.", "error")
            return render_template('index.html', 
                                   user_input_text_val=user_input_text, 
                                   extracted_text_preview=extracted_text_preview,
                                   summary_type_selected=summary_type_selected,
                                   extractive_word_count=extractive_word_count,
                                   abstractive_min_length=abstractive_min_length,
                                   abstractive_max_length=abstractive_max_length)


        if input_source == "manual":
            user_input_text = request.form.get('user_input_text', '').strip()
            if not user_input_text:
                flash("Mohon masukkan teks terlebih dahulu untuk diringkas.", "warning")
        elif input_source == "url":
            article_url = request.form.get('article_url_input', '').strip()
            if not article_url:
                flash("Mohon masukkan URL artikel.", "warning")
            elif not is_valid_url(article_url):
                flash("URL yang Anda masukkan tidak valid. Pastikan format URL benar (misalnya, dimulai dengan `http://` atau `https://`).", "warning")
            else:
                extracted_text = get_text_from_url(article_url)
                if extracted_text:
                    flash("Teks dari URL berhasil diambil!", "success")
                    user_input_text = extracted_text
                    extracted_text_preview = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                else:
                    flash("Gagal mengambil teks dari URL. Pastikan URL mengarah ke artikel yang dapat diproses dan tidak ada masalah jaringan.", "error")
                    user_input_text = ""
        
        if user_input_text and user_input_text.strip():
            if summary_type_selected == "Extractive (TextRank)":
                if word2vec_model is None:
                    flash("Model Word2Vec tidak dapat dimuat. Proses peringkasan TextRank tidak dapat dilakukan.", "error")
                else:
                    summary_result, summary_status = textrank_summarize(
                        user_input_text,
                        word2vec_model,
                        extractive_word_count
                    )
            elif summary_type_selected == "Abstractive (IndoBART)":
                if indobart_tokenizer is None or indobart_model is None:
                    flash("Model IndoBART tidak dapat dimuat. Proses peringkasan Abstractive tidak dapat dilakukan.", "error")
                else:
                    summary_result, summary_status = indobart_summarize(
                        user_input_text,
                        indobart_tokenizer,
                        indobart_model,
                        abstractive_min_length,
                        abstractive_max_length
                    )
            
            if summary_status == "Sukses":
                flash(f"Ringkasan berhasil dibuat! Jenis: {summary_type_selected}. Total Kata: {len(word_tokenize(summary_result))}", "success")
            else:
                flash(f"Gagal membuat ringkasan. Status: {summary_status}", "error")

    return render_template('index.html', 
                           user_input_text_val=user_input_text, 
                           extracted_text_preview=extracted_text_preview,
                           summary_result=summary_result,
                           summary_status=summary_status,
                           summary_type_selected=summary_type_selected,
                           extractive_word_count=extractive_word_count,
                           abstractive_min_length=abstractive_min_length,
                           abstractive_max_length=abstractive_max_length)

if __name__ == '__main__':
    # Ini hanya untuk menjalankan secara lokal.
    # Di produksi (misalnya di Render), Gunicorn akan menjalankan aplikasi.
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))