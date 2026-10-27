# Exercise 4.20 - Modifikasi Question Answering dengan Teks & Pertanyaan Baru
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "distilbert-base-cased-distilled-squad"

# Memuat tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Memilih potongan teks (konteks) dan pertanyaan yang berbeda
context = """
TensorFlow is a free and open-source software library for machine learning and artificial intelligence. 
It was developed by the Google Brain team and released in November 2015.
"""

questions = [
    "Who developed TensorFlow?",
    "When was TensorFlow released?"
]

# Melakukan inferensi untuk setiap pertanyaan
for question in questions:
    inputs = tokenizer(question, context, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits) + 1
    
    predict_answer_tokens = inputs["input_ids"][0][answer_start_index:answer_end_index]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")