import numpy as np
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from model_and_eval import FlexibleCNN

def image_to_pixel_string(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Flatten the array to a 1D array
    flat_array = image_array.flatten()

    # Convert the 1D array to a comma-separated string
    pixel_string = ','.join(map(str, flat_array))

    return pixel_string

def preprocess_single_image(pixel_string):
    # Convert the comma-separated string to a NumPy array
    pixel_array = np.fromstring(pixel_string, sep=',', dtype=int)

    # Reshape the array to match the input shape expected by the model
    image_tensor = torch.tensor(pixel_array.reshape(1, 1, 48, 48), dtype=torch.float32)

    return image_tensor

def predict_label(model_path, image_tensor):
    # Load the model
    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)

    # Set the model to evaluation mode
    loaded_model.eval()

    # Make a prediction for the single image
    with torch.no_grad():
        output = loaded_model(image_tensor)
        _, predicted_label = torch.max(output, 1)

    # Adjust the predicted label by adding 1
    predicted_label = predicted_label.item() + 1

    return predicted_label

def display_image_with_label(image_path, predicted_label):
    # Load the image
    image = Image.open(image_path)
    label_name = get_label(predicted_label)
    # Display the image with the predicted label
    plt.imshow(image)
    
    plt.title(f"Predicted Label: {label_name}")
    plt.show()
    
def get_label(predicted_label):
    if predicted_label == 1:
        return "Neutral"
    elif predicted_label == 2:
        return "Engaged/Focus"
    elif predicted_label == 3:
        return "Bored/looking Away"
    else:
        return "Angry"
    
if __name__ == "__main__":
    # Set the path to the saved main model
    main_model_path = "mainmodel.pkl"

    # Set the path to the image file
    image_path = 'Sample/engaged/13_.png'

    # Convert the image to a pixel string
    pixel_string = image_to_pixel_string(image_path)

    # Preprocess the single image
    image_tensor = preprocess_single_image(pixel_string)

    # Predict the label for the single image
    predicted_label = predict_label(main_model_path, image_tensor)

    # Display the image with the predicted label
    display_image_with_label(image_path, predicted_label)
