# Exercise 4.15 - Mengubah n = 1 menjadi n = 2 (Layer kedua: block1_conv2)
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

# Load model VGG19
model = VGG19()

# Ubah n menjadi 2 untuk mengakses layer block1_conv2
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Nama Layer:", model.layers[n].name)
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters: ", s[3])

# Normalize filter values to 0-1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plot first few filters
n_filters, ix = 4, 1

pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[j, :, :], cmap='gray')
        ix += 1

pyplot.show()