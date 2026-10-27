#Example 4.4 LeNet-5 Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

model = Sequential()

# Layer Konvolusi 1
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
# Layer Pooling 1 (Diperbaiki dengan pool_size)
model.add(AveragePooling2D(pool_size=(2, 2)))

# Layer Konvolusi 2
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
# Layer Pooling 2 (Diperbaiki dengan pool_size)
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