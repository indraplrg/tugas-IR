# Import Semua Package
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import streamlit as st

# Membaca Dataset
data = pd.read_json('News_Category_Dataset_v3.json', lines=True)  # Sesuaikan dengan nama file dataset
data = data[['headline', 'short_description']]  # Pilih kolom yang relevan
data['text'] = data['headline'] + " " + data['short_description']

# Trim dan Ubah ke huruf kecil
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

data['processed_text'] = data['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['processed_text'])  #

# Mencari Artikel sesuai kata kunci
def search_articles(query, tfidf_matrix, vectorizer, data, top_n=5):
    query_processed = preprocess_text(query)
    query_vector = vectorizer.transform([query_processed])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

st.title("Pencarian Artikel Berita Berdasarkan Kata Kunci")

query = st.text_input("Masukkan kata kunci pencarian:")
if query:
    results = search_articles(query, tfidf_matrix, vectorizer, data)
    st.write("Hasil pencarian:")
    for index, row in results.iterrows():
        st.write(f"**{row['headline']}**")
        st.write(f"{row['short_description']}")
        st.write("---")
