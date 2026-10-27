# Example 4.23 The Transformers.py Program
# https://github.com/huggingface/transformers
# pip install transformers

from transformers import pipeline
classifier = pipeline('sentiment-analysis')
result = classifier('This is a good movie.')
print(result)