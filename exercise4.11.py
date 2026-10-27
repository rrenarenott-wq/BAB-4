# EXERCISE 4.11 (VGG19 Webcam Classifier)
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
# MODIFIKASI: Mengubah import menjadi vgg19
from tensorflow.keras.applications import vgg19 
import numpy as np

image_size = 224
# MODIFIKASI: Memanggil VGG19 menggantikan VGG16
model = vgg19.VGG19(weights='imagenet')
print(model.summary())

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    # MODIFIKASI: Menggunakan fungsi preprocess_input bawaan vgg19
    processed_image = vgg19.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    label = decode_predictions(predictions)

    # format final image visualization to display the results of experiments
    # MODIFIKASI: Mengubah teks tampilan menjadi "VGG19:"
    cv2.putText(cam_frame, "VGG19: {}, {:.1f}".format(label[0][0][1], label[0][0][2]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27: # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()