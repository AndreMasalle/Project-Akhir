import re

# Rule-based filtering dictionary
RULES = {
    'website': ['laravel', 'django', 'react', 'next.js', 'angular', 'vue'],
    'mobile': ['kotlin', 'swift', 'flutter', 'jetpack compose'],
    'game/ar/vr': ['unity', 'unreal', 'c#', 'blender', 'arkit', 'vuforia'],
    'desktop': ['java', 'electron', 'qt', 'javafx', 'wpf'],
    'machine learning': ['python', 'tensorflow', 'pytorch', 'keras', 'scikit-learn'],
    'iot': ['nodemcu', 'esp', 'arduino', 'raspberry pi', 'micropython']
}

def extract_platforms(tech_str):
    """
    Ekstrak platform berdasarkan teknologi yang terdeteksi
    """
    platforms = []
    for tech in tech_str.split(','):
        for platform, keywords in RULES.items():
            if tech in keywords:
                platforms.append(platform)
    return list(set(platforms))


def ner_detect_platform_and_tech(text):
    """Deteksi platform dan teknologi dari teks menggunakan regex.

    Function ini memindai teks input untuk menemukan teknologi yang terdefinisi dalam `RULES`,
    lalu mengidentifikasi platform terkait berdasarkan mapping yang ada.

    Args:
        text (str): Teks mentah/input pengguna yang akan dianalisis (contoh: "Saya menggunakan Python di web").

    Returns:
        tuple:
            - list: Platform unik yang terdeteksi (contoh: ["Web", "Mobile"]).
            - list: Teknologi yang berhasil diidentifikasi dari teks (contoh: ["python", "flutter"]).

    Example:
        > ner_detect_platform_and_tech("Aplikasi dibangun dengan React Native dan Firebase")
        (['Mobile', 'Cloud'], ['react native', 'firebase'])

    Note:
        - Pencarian bersifat **case-insensitive** (termasuk huruf kecil/besar).
        - Menggunakan `word boundaries` untuk menghindari partial match (misal: "java" tidak akan match dengan "javascript").
        - Jika teknologi muncul di beberapa platform di `RULES`, platform akan dikumpulkan untuk semua kemunculan.
    """
    if not isinstance(text, str):
        return [], []

    detected_tech = []
    text_lower = text.lower()

    for platform, tech_list in RULES.items():
        for tech in tech_list:
            if re.search(rf'\b{re.escape(tech)}\b', text_lower):
                detected_tech.append(tech)

    detected_platforms = []
    for tech in detected_tech:
        for platform, tech_list in RULES.items():
            if tech in tech_list:
                detected_platforms.append(platform)

    return list(set(detected_platforms)), detected_tech
