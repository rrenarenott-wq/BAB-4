# EXERCISE 4.9 (Memuat Model VGG19)
from tensorflow.keras.applications import vgg19

# Inisialisasi model VGG19 dengan bobot pre-trained ImageNet
model = vgg19.VGG19(weights='imagenet')

# Menampilkan ringkasan arsitektur model
print(model.summary())