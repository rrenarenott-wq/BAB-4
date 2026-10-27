# Exercise 4.4 LeNet-5 Keras (Modifikasi Jumlah Filter)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

model = Sequential()

# Layer Konvolusi 1 
# MODIFIKASI: Mengubah filters dari 6 menjadi 12
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(AveragePooling2D(pool_size=(2, 2)))

# Layer Konvolusi 2 
# MODIFIKASI: Mengubah filters dari 16 menjadi 32
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

# Meratakan matriks menjadi vektor
model.add(Flatten())

# Fully Connected Layer (Dense)
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))

# Output Layer
model.add(Dense(units=10, activation='softmax'))

# Menampilkan ringkasan arsitektur model
model.summary()