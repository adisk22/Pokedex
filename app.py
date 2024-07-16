import cv2
from data_handler import load_data, get_pokemon_by_name
from image_processing import load_image, preprocess_image
from model import create_model, load_model, setup_model
import os


def predict_pokemon(image_path, model, data):
    try:
        img = load_image(image_path)  # Assumes function to load and preprocess the image
        img_preprocessed = preprocess_image(img)  # Assumes function to preprocess the image
        prediction = model.predict(img_preprocessed[None, ...])  # Add batch dimension
        predicted_index = min(prediction.argmax(), 152)  # Get the index of the highest probability
        predicted_name = data['name'][predicted_index + 1]  # Retrieve the name from the DataFrame
    except Exception as e:
        print(f"An error occurred: {e}")
        predicted_name = "Unknown"  # Default value in case of any failure

    return predicted_name

    


def main():

    data = load_data("data/PokeInfo.csv")
    model = setup_model()

    image_path = r"data\PokeImages\Vileplume\0.jpg"

    result = predict_pokemon(image_path, model, data)
    print("Predicted Pokemon:", result)

if __name__ == "__main__":
    main()

