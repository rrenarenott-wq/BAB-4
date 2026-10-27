# Exercise 4.16 - Menampilkan seluruh filter dengan n_filters = s[3]
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

# Load model VGG19
model = VGG19()

n = 1  # Bisa disesuaikan (misal layer pertama)
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Total number of filters: ", s[3])

# Normalize filter values to 0-1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Mengubah jumlah filter yang ditampilkan menjadi seluruh total filter (s[3] = 64)
n_filters, ix = s[3], 1

pyplot.figure(figsize=(12, 12))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[j, :, :], cmap='gray')
        ix += 1

pyplot.show()