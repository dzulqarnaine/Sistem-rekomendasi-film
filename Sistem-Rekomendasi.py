import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# --- PINDAHKAN INI KE ATAS ---
st.set_page_config(
    page_title="Sistem Rekomendasi Film",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --------------------------------

# --- 1. KONFIGURASI & DEFINISI ARSITEKTUR ---

MODEL_NAME = 'bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CombinedModel(nn.Module):
    def __init__(self, model_name, num_labels, num_non_text_features):
        super(CombinedModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.bert.config.hidden_size + num_non_text_features, num_labels))
    def forward(self, input_ids, attention_mask, non_text_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_output.last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_embedding, non_text_features], dim=1)
        return self.classifier(combined_features)

# --- 2. FUNGSI UNTUK MEMUAT ASET (DENGAN CACHE) ---

@st.cache_data
def load_data():
    df = pd.read_csv('dataset_final_lengkap.csv')
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    embeddings = np.load('movie_embeddings.npy')
    return df, embeddings

@st.cache_resource
def load_model_and_dependencies():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)

    num_labels = len(mlb.classes_)
    num_non_text_features = preprocessor.named_transformers_['cat'].get_feature_names_out().shape[0] + 2

    model = CombinedModel(MODEL_NAME, num_labels, num_non_text_features)
    model.load_state_dict(torch.load("best_model_state.bin", map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, preprocessor, mlb

# --- 3. PEMUATAN UTAMA ---
df_final, movie_embeddings = load_data()
model, tokenizer, preprocessor, mlb = load_model_and_dependencies()

# --- 4. FUNGSI HELPER ---
def get_bert_embedding(text, bert_model, tokenizer):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=256)
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    return model_output.last_hidden_state[:, 0, :].cpu().numpy()

indices_map = pd.Series(df_final.index, index=df_final['movie_title'])

# --- 5. TAMPILAN APLIKASI ---
with st.sidebar:
    selected = option_menu(menu_title="Menu Utama", options=["Overview", "Rekomendasi Film"], icons=["house", "film"], menu_icon="cast", default_index=0)

def halaman_overview():
    st.markdown("<h1 style='text-align: center; color: skyblue;'>Rekomendasi Film Adaptif</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: yellow;'>Data Analysis Competition IFest 2025</h4>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Tentang Proyek")
    st.markdown("""<p style='text-align: justify;'>Di tengah melimpahnya pilihan film di berbagai platform digital, pengguna seringkali mengalami kesulitan dalam menemukan tontonan yang sesuai dengan selera mereka. Proyek ini bertujuan untuk mengatasi masalah tersebut dengan membangun sebuah <b>Sistem Rekomendasi Film Cerdas</b>.<br><br>Kami menggunakan pendekatan <i>Content-Based Filtering</i> dengan memanfaatkan model <i>deep learning</i> canggih, <b>BERT</b>, untuk memahami makna dan konteks dari sinopsis setiap film. Dengan mengubah setiap cerita menjadi 'profil makna' numerik, kami dapat secara akurat mengukur kemiripan antar film dan memberikan rekomendasi yang relevan.<br><br>Berdasarkan pengujian offline yang ketat, sistem kami berhasil mencapai tingkat presisi yang sangat tinggi, yaitu <b>Precision@10 sebesar 95.38%</b>, membuktikan kemampuannya dalam menyajikan rekomendasi yang berkualitas.</p>""", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1574267432553-4b4628081c31?q=80&w=1931&auto=format&fit=crop", caption="Data-Driven Movie Recommendations")

def halaman_rekomendasi():
    st.markdown("<h1 style='text-align: center;'>Rekomendasi Film</h1>", unsafe_allow_html=True)
    st.write("Temukan film favorit Anda berikutnya! Pilih film dari database kami atau masukkan detail film baru yang Anda suka.")

    tab1, tab2 = st.tabs(["Rekomendasi dari Database", "Rekomendasi untuk Film Baru"])

    with tab1:
        st.subheader("Pilih Film dari Database Kami")
        title_list = df_final['movie_title'].tolist()
        selected_title = st.selectbox("Pilih judul film:", options=title_list)

        if st.button("Dapatkan Rekomendasi dari Database"):
            idx = indices_map[selected_title]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            
            movie_embedding = movie_embeddings[idx].reshape(1, -1)
            sim_scores = cosine_similarity(movie_embedding, movie_embeddings)[0]

            # --- PERBAIKAN LOGIKA ---
            # Ambil 11 film teratas, lalu buang yang pertama (film itu sendiri)
            top_indices = sim_scores.argsort()[-11:][::-1]
            top_indices = top_indices[1:] # Buang film pertama
            
            rekomendasi = df_final['movie_title'].iloc[top_indices]
            
            st.success(f"Rekomendasi film yang mirip dengan '{selected_title}':")
            for i, film in enumerate(rekomendasi):
                st.write(f"{i+1}. {film}")

    with tab2:
        st.subheader("Masukkan Detail Film Baru yang Anda Suka")
        judul_baru = st.text_input("Judul Film Baru:")
        sinopsis_baru = st.text_area("Sinopsis Film Baru:")

        if st.button("Dapatkan Rekomendasi untuk Film Baru"):
            if judul_baru and sinopsis_baru:
                with st.spinner("Menganalisis film Anda dan mencari rekomendasi..."):
                    bert_model = model.bert
                    embedding_baru = get_bert_embedding([sinopsis_baru], bert_model, tokenizer)
                    
                    sim_scores = cosine_similarity(embedding_baru, movie_embeddings)[0]
                    
                    # --- PERBAIKAN LOGIKA ---
                    # Ambil 11 film teratas untuk cadangan
                    top_indices = sim_scores.argsort()[-11:][::-1]
                    
                    # Ambil judul-judul film yang direkomendasikan
                    recommended_titles = df_final['movie_title'].iloc[top_indices]

                    # Jika film teratas adalah film input itu sendiri, buang film tersebut
                    if recommended_titles.iloc[0].lower() == judul_baru.lower():
                        rekomendasi_final = recommended_titles.iloc[1:]
                    else:
                        # Jika tidak, ambil 10 teratas
                        rekomendasi_final = recommended_titles.head(10)
                    
                    st.success(f"Rekomendasi film yang mirip dengan '{judul_baru}':")
                    for i, film in enumerate(rekomendasi_final):
                        st.write(f"{i+1}. {film}")
            else:
                st.warning("Mohon masukkan judul dan sinopsis film baru.")

# --- 6. MENJALANKAN APLIKASI ---
if selected == "Overview":
    halaman_overview()
elif selected == "Rekomendasi Film":
    halaman_rekomendasi()