# Example 4.17 - Complete Feature Map Visualization (Part 1 to Final)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# ==========================================
# BAGIAN 1: Memuat Model ResNet50
# ==========================================
# Memuat model praterlatih ResNet50 tanpa layer klasifikasi bagian atas (include_top=False)
model = ResNet50(weights='imagenet', include_top=False)
print("Model ResNet50 berhasil dimuat.")
print(model.summary())

# ==========================================
# BAGIAN 2: Memuat dan Mempersiapkan Gambar
# ==========================================
# Menyiapkan gambar uji (misalnya 'Elephant.jpg')
img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))

# Mengubah gambar menjadi array numpy
x = image.img_to_array(img)

# Menambah dimensi batch (menjadi bentuk (1, 224, 224, 3))
x = np.expand_dims(x, axis=0)

# Melakukan preprocessing gambar sesuai kebutuhan ResNet50
x = preprocess_input(x)

# ==========================================
# BAGIAN 3: Ekstraksi Fitur dari Layer Tertentu
# ==========================================
# Mengambil keluaran dari layer konvolusi tertentu (misalnya layer ke-1 atau layer spesifik)
# Kita bisa membuat sub-model yang mengambil input gambar dan memberikan output dari layer target
layer_name = "conv2_block1_out"  # Contoh nama layer konvolusi di ResNet50
extractor = Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)

# Mendapatkan feature maps hasil prediksi/ekstraksi
feature_maps = extractor.predict(x)
print(f"Bentuk (Shape) dari Feature Map: {feature_maps.shape}")

# ==========================================
# BAGIAN 4 & 5: Visualisasi Feature Maps
# ==========================================
# Feature maps biasanya berdimensi (1, tinggi, lebar, jumlah_channel)
# Kita ambil channel pertama atau beberapa channel pertama untuk divisualisasikan
square = 8  # Jumlah kolom/baris grid visualisasi
ix = 1

plt.figure(figsize=(12, 12))
for _ in range(square):
    for _ in range(square):
        # Menentukan subplot
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Menampilkan channel filter ke- (ix - 1) dari feature map
        # feature_maps[0, :, :, ix-1] mengambil matriks 2D dari channel tertentu
        plt.imshow(feature_maps[0, :, :, ix - 1], cmap='viridis')
        ix += 1

# ==========================================
# BAGIAN 6: Menampilkan Plot Akhir
# ==========================================
plt.suptitle(f"Feature Maps dari Layer: {layer_name}", fontsize=16)
plt.tight_layout()
plt.show()