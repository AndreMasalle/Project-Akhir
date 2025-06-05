from flask import Flask, request, jsonify
# from flask_sqlalchemy import SQLAlchemy
import numpy as np
import faiss
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import logging
from Preprocess import preprocessing, preprocess_list_teknologi
from RBF import ner_detect_platform_and_tech

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Inisialisasi variabel global
sbert_model = None
faiss_index = None
processed_data = None


def load_models():
    """Memuat model FAISS, data terproses, dan model Sentence Transformers.

    Returns:
        bool: True jika semua model berhasil dimuat, False jika terjadi error.
    """
    global sbert_model, faiss_index, processed_data
    try:
        model_path = "./models"
        if not os.path.exists(model_path):
            logger.error("Model directory not found")
            return False

        # Load model
        faiss_index = faiss.read_index(os.path.join(model_path, "faiss_index.bin"))
        processed_data = pd.read_pickle(os.path.join(model_path, "processed_data.pkl"))
        sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


@app.route('/api/search', methods=['POST'])
def search():
    """Endpoint untuk pencarian similarity dengan 4 input utama."""

    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        judul_pa = data.get('judul_pa', '')
        desc_pa = data.get('desc_pa', '')
        platform_aplikasi = data.get('platform_aplikasi', '')
        teknologi_yg_digunakan = data.get('teknologi_yg_digunakan', [])
        threshold = float(data.get('threshold', 0.1))

        # Validasi input wajib
        if not judul_pa and not desc_pa:
            return jsonify({"error": "Judul dan deskripsi tidak boleh kosong"}), 400

        if not teknologi_yg_digunakan:
            return jsonify({"error": "Teknologi tidak boleh kosong"}), 400

        # Gabungkan judul dan deskripsi, lalu preprocessing
        query_text = f"{judul_pa} {desc_pa}"
        processed_query = preprocessing(query_text)
        processed_teknologi = preprocess_list_teknologi(teknologi_yg_digunakan)

        # Deteksi NER
        detected_platforms_ner, detected_techs_ner = ner_detect_platform_and_tech(query_text)

        # Embedding dan normalisasi
        query_vector = sbert_model.encode([processed_query], convert_to_tensor=False).astype(np.float32)
        faiss.normalize_L2(query_vector)

        # FAISS search
        distances, indices = faiss_index.search(query_vector, 50)

        # Ambil data sekaligus dari processed_data
        matched_rows = processed_data.iloc[indices[0]].to_dict(orient='records')

        results = []
        for i, row in enumerate(matched_rows):
            score = distances[0][i]
            if score > threshold:
                results.append({
                    'data_pa_id': int(row['data_pa_id']),
                    'judul_pa': row['judul_pa'],
                    'similarity_score': round(float(score) * 100, 2),
                    'platform': row['platform_aplikasi'],
                    'kategori': row['kategori'],
                    'teknologi': row['teknologi_yg_digunakan'],
                    'tahun_ajaran': row['tahun_ajaran'],
                    'dosen_pembimbing': row['dosen_pembimbing'],
                    'mahasiswa': row['mahasiswa']
                })

        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:5]

        return jsonify({
            "results": results,
            "detected_platforms": detected_platforms_ner,
            "detected_techs": detected_techs_ner
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": faiss_index is not None
    })


if __name__ == '__main__':
    load_models()
    app.run(port=5000)