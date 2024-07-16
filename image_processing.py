import cv2
import os

def load_image(image_path): # Loads an image from the file
    return cv2.imread(image_path)

def preprocess_image(image): # Preprocessing image for model prediction
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

if __name__ == "__main__":
    img = load_image("data/PokeImages/Abra/0.jpg")
    img_preprocessed = preprocess_image(img)
    cv2.imshow("Processed Image", img_preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    