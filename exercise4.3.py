# Exercise 4.3 Multiple Layer Perceptron Classification (Modified to 5 neurons)
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time # Tambahan library untuk menghitung waktu

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 7)

features = X_train.shape[1]

#MPL 3-layer Model ========================================================
model = Sequential()
# MODIFIKASI EXERCISE 4.3: Mengubah parameter 10 menjadi 5 pada Dense layer
model.add(Dense(5, activation='relu', input_shape=[X_train.shape[1]]))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

print(model.summary())
print(model.get_config())

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

# Mulai menghitung waktu training
start_time = time.time()

model.fit(X_train, y_train, batch_size = 50, validation_split=0.2, epochs = 100, verbose = 1)

# Berhenti menghitung waktu training
end_time = time.time()

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('accuracy: ', results[1])
print(f'Total training time: {end_time - start_time:.2f} seconds')