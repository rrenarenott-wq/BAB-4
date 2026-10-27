# Exercise 4.19 - Analisis Sentimen dengan Berbagai Kalimat (Positif & Negatif)
from transformers import pipeline

# Inisialisasi pipeline analisis sentimen
classifier = pipeline('sentiment-analysis')

# Daftar kalimat uji yang berisi campuran sentimen positif dan negatif
sentences = [
    "I absolutely love this product, it exceeded all my expectations!",
    "This was a complete waste of money and time. Terrible experience.",
    "The service was okay, nothing special to write home about.",
    "Worst customer support I have ever encountered in my life."
]

# Melakukan prediksi sentimen untuk setiap kalimat
for text in sentences:
    result = classifier(text)
    print(f"Text: '{text}'")
    print(f"Result: {result}\n")