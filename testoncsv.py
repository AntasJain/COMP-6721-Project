import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model_and_eval import FlexibleCNN 

def load_and_preprocess_data(test_csv_path):
    # Load the test CSV file
    test_data = pd.read_csv(test_csv_path)

    # Preprocess the data as needed 
    test_data['pixels'] = test_data['pixels'].apply(lambda x: np.fromstring(x, sep=',', dtype=int))

    X_test = np.array(test_data['pixels'].tolist())
    X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 48, 48), dtype=torch.float32)

    return X_test_tensor, test_data['emotion']

def run_mainmodel_on_test(model_path, test_data_tensor, actual_labels):
    # Load the main model 
    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)
    # Set the model to evaluation mode
    loaded_model.eval()
    # Create a DataLoader for the test data
    batch_size = 32  # Adjust as needed
    test_loader = DataLoader(test_data_tensor, batch_size=batch_size, shuffle=False)

    # Perform predictions on the test data
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            outputs = loaded_model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Adjust predicted labels by adding 1
    predictions = np.array(predictions) + 1

    # Calculate and print the accuracy
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"Test Accuracy: {accuracy}")

    # Save or use the predictions as needed
    result_df = pd.DataFrame({'Actual_Label': actual_labels, 'Predicted_Label': predictions})
    result_df.to_csv("mainmodel_predictions.csv", index=False)

if __name__ == "__main__":
    # Set the path to  test CSV file
    test_csv_path = "test.csv"

    # Load and preprocess the test data
    X_test_tensor, actual_labels = load_and_preprocess_data(test_csv_path)

    # Set the path to saved main model
    main_model_path = "mainmodel.pkl"

    # Run the main model on the test data and print the accuracy
    run_mainmodel_on_test(main_model_path, X_test_tensor, actual_labels)