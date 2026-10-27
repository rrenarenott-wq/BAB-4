# EXERCISE 4.13 & 4.14 (Diperbaiki tanpa menggunakan library usang image-classifiers)
import numpy as np
import cv2
from keras.applications.imagenet_utils import decode_predictions

# Memuat arsitektur dan fungsi preprocessing resmi dari bawaan TensorFlow Keras
from tensorflow.keras.applications import (
    VGG16, vgg16,
    ResNet50, resnet50,
    MobileNetV2, mobilenet_v2,
    DenseNet201, densenet,
    InceptionV3, inception_v3
)

# Ubah nilai teks pada variabel ini untuk mengganti model yang diuji:
# Pilihan: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', 'inceptionv3'
nama_model = 'mobilenetv2' 

# Penerapan blok if-else untuk pemilihan model secara aman
if nama_model == 'vgg16':
    model = VGG16(weights='imagenet')
    preprocess_input = vgg16.preprocess_input
    sz = 224
elif nama_model == 'resnet50':
    model = ResNet50(weights='imagenet')
    preprocess_input = resnet50.preprocess_input
    sz = 224
elif nama_model == 'mobilenetv2':
    model = MobileNetV2(weights='imagenet')
    preprocess_input = mobilenet_v2.preprocess_input
    sz = 224
elif nama_model == 'densenet201':
    model = DenseNet201(weights='imagenet')
    preprocess_input = densenet.preprocess_input
    sz = 224
elif nama_model == 'inceptionv3':
    model = InceptionV3(weights='imagenet')
    preprocess_input = inception_v3.preprocess_input
    sz = 299
else:
    print("Error: Nama model tidak dikenali.")
    exit()

print(f"Sedang memuat arsitektur {nama_model}...")

camera = cv2.VideoCapture(0)

while True:
    ret, cam_frame = camera.read()
    if not ret:
        break
        
    frame = cv2.resize(cam_frame, (sz, sz))
    
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    # verbose=0 agar terminal tidak dipenuhi teks loading pada setiap frame
    preds = model.predict(image, verbose=0) 
    label = decode_predictions(preds)

    cv2.putText(cam_frame, f"{nama_model}: {label[0][0][1]} ({label[0][0][2]:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Classification", cam_frame)
    
    key = cv2.waitKey(30)
    if key == 27: # Tekan 'ESC' untuk keluar
        break

camera.release()
cv2.destroyAllWindows() 