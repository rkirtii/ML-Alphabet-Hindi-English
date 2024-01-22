from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

model = keras.models.load_model("My_model.h5")

class_to_alphabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 

                    26: 'z_Kaa', 27: 'z_Kha', 28: 'z_Gaa', 29: 'z_Gha', 30: 'z_Kna', 31: 'z_Cha', 32: 'z_Chha', 33: 'z_Ja',
                    34: 'z_Jha', 35: 'z_Yna', 36: 'z_Tamatar', 37: 'z_Tha', 38: 'z_Dholak-Daa', 39: 'z_Dha', 40: 'z_Adna', 41: 'z_Taa',
                    42: 'z_Tha', 43: 'z_Dada_wala_da', 44: 'z_Dhaa', 45: 'z_Naa', 46: 'z_Paa', 47: 'z_Faa', 48: 'z_Baa', 49: 'z_Bha',
                    50: 'z_Maa', 51: 'z_Yaa', 52: 'z_Raa', 53: 'z_Laa', 54: 'z_Waa', 55: 'z_Sha', 56: 'z_Shadyantra',
                    57: 'z_Saa', 58: 'z_Haa', 59: 'z_Chhya', 60: 'z_Traa', 61: 'z_Gyaa'}


# Predicting results
dir_path = 'basedata/test/'
for i in os.listdir(dir_path):
    img_path = os.path.join(dir_path, i)
    
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 128, 128, 3))  # Assuming color images with 3 channels
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = int(predictions.argmax(axis=-1))

    predicted_alphabet = class_to_alphabet[predicted_class]

    plt.imshow(img)
    plt.title(f"Predicted Alphabet: {predicted_alphabet}")
    plt.show()
