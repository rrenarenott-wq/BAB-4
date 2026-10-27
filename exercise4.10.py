# EXERCISE 4.10 (Klasifikasi Gambar dengan VGG19)
from tensorflow.keras.preprocessing import image
# MODIFIKASI: Mengubah vgg16 menjadi vgg19
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

# MODIFIKASI: Memanggil VGG19 menggantikan VGG16
model = VGG19(weights='imagenet')

# Pastikan file 'Elephant.jpg' yang kamu siapkan sebelumnya masih ada di folder
img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Preprocessing data gambar sesuai standar VGG19
x = preprocess_input(x)

# prediction
predictions = model.predict(x)
results = decode_predictions(predictions)

print("Hasil Prediksi VGG19:")
print(results)