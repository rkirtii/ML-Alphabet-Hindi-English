from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

train_data = ImageDataGenerator(rescale=1/255)
Train_label = ImageDataGenerator(rescale=1/255)

Train_Images = train_data.flow_from_directory("./basedata/train_images", class_mode='sparse', target_size=(128, 128))
Train_Labels = Train_label.flow_from_directory("./basedata/train_labels", class_mode='sparse', target_size=(128, 128))


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(62, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


training = model.fit(Train_Images, epochs=6, validation_data=Train_Labels)


model.save("My_model.h5")

# dir_path='basedata/test/'
# for i in os.listdir(dir_path):
#     img=image.load_img(dir_path + i)
#     plt.imshow(img)
#     plt.show()