import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle



image_path = "resources/engaged_images/0004.jpg"
input_image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Applying the transformation to the input image
input_tensor = transform(input_image)
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

# Load  main model
with open("mainmodel.pkl", "rb") as file:
    loaded_model1 = pickle.load(file)
loaded_model1.eval()  # Set the model to evaluation mode

# Making the prediction
with torch.no_grad():
    output = loaded_model1(input_tensor)

# Get the predicted class index
_, predicted_class = torch.max(output, 1)

class_labels = ["Neutral", "Engaged/Focused", "Bored/Looking Away", "Angry"]
predicted_label = class_labels[predicted_class.item()]

# Print the predicted label
print("Predicted Label:", predicted_label)

plt.imshow(input_image, cmap='gray')
plt.title(f'Predicted Label: {predicted_label}')
plt.axis('off')
plt.show()