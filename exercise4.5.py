# Exercise 4.5 AlexNet with Keras (Modifikasi Dense Layer)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

model = Sequential()

# Layer Konvolusi (Tetap sama seperti Example 4.5)
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), activation='relu', strides=(4,4), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters=256, kernel_size=(11,11), activation='relu', strides=(1,1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Flatten())

# 1st Fully Connected layer 
# MODIFIKASI: Mengubah unit dari 4096 menjadi 2048
model.add(Dense(2048, activation='relu', input_shape=(224*224*3,)))
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
# MODIFIKASI: Mengubah unit dari 4096 menjadi 2048
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
# MODIFIKASI: Mengubah unit dari 1000 menjadi 512
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(17,activation='softmax'))

# Cek perubahan jumlah parameter
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])