from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from functools import lru_cache
from ast import literal_eval


stemmer = StemmerFactory().create_stemmer()
stopwords = StopWordRemoverFactory().get_stop_words()


@lru_cache(maxsize=1024)
def preprocessing(teks):
    """
    Function ditujukan untuk melakukan preprocessing pada data teks seperti
    1. judul_pa
    2. desc_pa

    Berikut adalah langkah-langkahnya:
    1. Lowercase
    2. Tokenisasi
    3. Hapus non-alnum
    4. Hapus stopwords
    5. Stemming

    Args:
        teks (str): Teks input.

    Returns:
        str: Teks yang sudah dibersihkan

    Requirements:
        stopwords: Daftar stopword bahasa indonesia
        `from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory`

        stemming: Objek stemmer sastrawi
        `from Sastrawi.Stemmer.StemmerFactory import StemmerFactory`
    """

    if not isinstance(teks, str):
        return ""

    teks = teks.lower()
    tokens = teks.split()
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def preprocess_list_teknologi(list_teknologi):
    """
    Args:
        list_teknologi (str or list): Data `teknologi_yg_digunakan` dalam bentuk string list atau list Python

    Returns:
        str: Data `teknologi_yg_digunakan` yang sudah diproses, seperti 'kotlin,firebase'
    """
    if isinstance(list_teknologi, str):
        try:
            list_teknologi = literal_eval(list_teknologi)
        except Exception:
            return ""

    if not isinstance(list_teknologi, list):
        return ""

    processed_list = [tech.lower() for tech in list_teknologi if isinstance(tech, str)]

    return ",".join(processed_list)



