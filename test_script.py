import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('handwritten.keras')

image_number = 1

while image_number <= 9:
    try:
        img = cv2.imread(f"digits/digit_{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print(f"\nERROR FOR DIGIT {image_number}\n")
    finally:
        image_number += 1

