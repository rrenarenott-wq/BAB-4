# Exercise 4.18 - Modifikasi NLTK dengan Teks Berbeda
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('punkt_tab')

# Memilih potongan teks yang berbeda (misal tentang Machine Learning)
sentence = """
Machine learning (ML) is a branch of artificial intelligence (AI) focused on building 
applications that learn from data and improve their accuracy over time without being 
explicitly programmed. Data scientists at Google and Microsoft develop advanced neural 
networks to solve complex problems in computer vision and natural language processing.
"""

# Proses Tokenisasi
tokens = nltk.word_tokenize(sentence)
print("--- TOKENS ---")
print(tokens)

# Proses POS Tagging (Part-of-Speech)
tagged = nltk.pos_tag(tokens)
print("\n--- POS TAGGED ---")
print(tagged)

# Proses Named Entity Chunking
entities = nltk.chunk.ne_chunk(tagged)
print("\n--- NAMED ENTITIES ---")
print(entities)