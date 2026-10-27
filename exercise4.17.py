# Exercise 4.17 - Visualisasi Feature Maps pada Layer Konvolusi Lain (n = 2 atau n = 4)
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

# 1. Fungsi untuk plot feature maps
def plot_feature_maps(feature_maps):
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1
    
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
            
    plt.show()

# 2. Load model VGG19
model = VGG19()

# 3. Ubah nilai n sesuai layer yang ingin diamati (misal: n = 2 untuk block1_conv2 atau n = 4 untuk block2_conv1)
n = 2  # Kamu bisa ganti ke 4, 5, atau nomor layer konvolusi lainnya

# Redefine model to output right after the selected hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
print(f"Menampilkan feature maps untuk layer index {n}: {model.layers[n].name if hasattr(model.layers[n], 'name') else 'layer'}")
model.summary()

# 4. Load dan proses gambar
img = load_img('Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# 5. Ekstraksi feature maps dan plot
feature_maps = model.predict(img)
print("Feature maps shape: ", feature_maps.shape)
plot_feature_maps(feature_maps)