# Example 4.24 The Transformers2.py Program (Manual Inference)
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "distilbert-base-cased-distilled-squad"

# Memuat tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = "What is the name of the company?"
context = "We created Biox Systems Ltd company back in the year of 2000."

# Tokenisasi input pertanyaan dan konteks
inputs = tokenizer(question, context, return_tensors="pt")

# Melakukan inferensi model
with torch.no_grad():
    outputs = model(**inputs)

# Mengambil token awal dan akhir jawaban
answer_start_index = torch.argmax(outputs.start_logits)
answer_end_index = torch.argmax(outputs.end_logits) + 1

predict_answer_tokens = inputs["input_ids"][0][answer_start_index:answer_end_index]
answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

# Menampilkan hasil dengan format mirip dictionary aslinya
print({'answer': answer, 'start': answer_start_index.item(), 'end': answer_end_index.item()})