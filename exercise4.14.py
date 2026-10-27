# Exercise 4.14 - Menggunakan if-else untuk memilih model klasifikasi
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

# Tentukan pilihan model yang ingin digunakan
model_name = "vgg19"  # Kamu bisa ubah menjadi "vgg16" jika ingin menggunakan VGG16

# Menggunakan if-else statement untuk memilih model
if model_name == "vgg16":
    print("Memuat model VGG16...")
    model = VGG16()
elif model_name == "vgg19":
    print("Memuat model VGG19...")
    model = VGG19()
else:
    print("Model tidak dikenali. Default menggunakan VGG16.")
    model = VGG16()

# Menampilkan ringkasan model yang dipilih
model.summary()