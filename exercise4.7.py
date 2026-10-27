# EXERCISE 4.7 (Modifikasi Filter dan Kernel)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from keras.utils import to_categorical

# Load & Preprocess Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create model
model = Sequential()
# MODIFIKASI EXERCISE 4.7: filter diubah ke 64, kernel diubah ke (3, 3)
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile & Tampilkan Summary
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("--- SUMMARY EXERCISE 4.7 ---")
print(model.summary())

# Fit & Evaluate
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))