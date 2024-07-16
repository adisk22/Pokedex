import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from custom_datasets import PyDataset
#dataset = PyDataset()
import os

def create_model(): ## Simple CNN model for classifying pokemon
     model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(151, activation='softmax')  # Assume 10 classes
    ])
     
     model.compile(optimizer ='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])
     return model

def prepare_data():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # Assuming your images are stored in a directory structure where each class has its own folder
    train_generator = train_datagen.flow_from_directory(
            'data/PokeImages',  # This is the target directory
            target_size=(224, 224),  # All images will be resized to 224x224
            batch_size=32,
            class_mode='categorical')  # Since we use categorical_crossentropy loss, we need categorical labels

    return train_generator

def setup_model():
    if os.path.exists("final_model.h5"):
        model = load_model("final_model.h5")
    
    else:
        model = train_model()

    return model

def train_model():
     model = create_model()
     train_generator = prepare_data()
     history = model.fit(train_generator, epochs=10, steps_per_epoch=100)
     model.save("final_model.h5")
     return model

if __name__ == "__main__":
     model = create_model()
     model.summary()
     