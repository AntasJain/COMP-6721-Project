from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class FlexibleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, num_conv_layers=2, kernel_size=7, dropout_rate=0.5):
        super(FlexibleCNN, self).__init__()

        # Define a list to hold the convolutional layers
        self.conv_layers = nn.ModuleList()

        # Add convolutional layers based on the specified number
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, 32, kernel_size=kernel_size, stride=1, padding=1))
            in_channels = 32  # Update in_channels for subsequent layers
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Dynamic calculation of the linear layer input size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.calculate_conv_output_size((1, 48, 48)), 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc2(x)
        return x

    def calculate_conv_output_size(self, input_shape):
        # Forward pass to get the shape
        with torch.no_grad():
            x = torch.randn(1, *input_shape)
            for layer in self.conv_layers:
                x = layer(x)
            x = self.flatten(x)
            return x.shape[1]

def split_test(data):
    """
    Function to split the validation and train data from general train file.
    """
    # Split the data into training and validation sets (80% train, 20% validation)
    train_data, validation_data = train_test_split(data, test_size=0.3, random_state=42)
    
    # Save the split datasets to separate CSV files
   
    validation_data2, test_data = train_test_split(validation_data, test_size=0.5, random_state=42)
    train_data['pixels'] = train_data['pixels'].apply(lambda x: ','.join(map(str, x)))
    train_data.to_csv("train.csv", index=False)
    
    validation_data2['pixels'] = validation_data2['pixels'].apply(lambda x: ','.join(map(str, x)))
    validation_data2.to_csv("val.csv", index=False)
    
    test_data['pixels'] = test_data['pixels'].apply(lambda x: ','.join(map(str, x)))
    test_data.to_csv("test.csv", index=False)

    print("Splitting completed and saved to train.csv and val.csv")
        
    return train_data,validation_data2,test_data

def get_confusion_matrix(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)

def calculate_metrics(actual, predicted):
    macro_precision = precision_score(actual, predicted, average='macro')
    macro_recall = recall_score(actual, predicted, average='macro')
    macro_f1 = f1_score(actual, predicted, average='macro')

    micro_precision = precision_score(actual, predicted, average='micro')
    micro_recall = recall_score(actual, predicted, average='micro')
    micro_f1 = f1_score(actual, predicted, average='micro')

    accuracy = accuracy_score(actual, predicted)

    return {
        'Macro Precision': macro_precision,
        'Macro Recall': macro_recall,
        'Macro F1 Score': macro_f1,
        'Micro Precision': micro_precision,
        'Micro Recall': micro_recall,
        'Micro F1 Score': micro_f1,
        'Accuracy': accuracy
    }    

def main():
        
    final_df = pd.read_csv("finalized_data.csv")
    final_df['pixels'] = final_df['pixels'].apply(lambda x: np.fromstring(x, sep=',', dtype=int)) # Convert string back to list or array
    length_of_image_arrays = final_df['pixels'].str.len()
    min_length = length_of_image_arrays.min()
    max_length = length_of_image_arrays.max()
    print(max_length,min_length)
    train_data, val_data, test_data = split_test(final_df)

    print(final_df.info())
    
    X = np.array(final_df['pixels'].tolist())
    y = np.array(final_df['emotion'])
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
    
    print("Train Data:")
    unique_values, counts = np.unique(y_train, return_counts=True)
    print("Classes", unique_values)
    print("Counts", counts)
    
    print("Test Data:")
    unique_values, counts = np.unique(y_test, return_counts=True)
    print("Classes", unique_values)
    print("Counts", counts)
    
    print("validation Data:")
    unique_values, counts = np.unique(y_val, return_counts=True)
    print("Classes", unique_values)
    print("Counts", counts)
    
    results_df = pd.DataFrame(columns=["Convolutional_Layer_Size", "Kernel_Size", "Epoch_Size", "Test_Accuracy", "Train_Accuracy"])

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 48, 48), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train-1, dtype=torch.long)
    
    X_val_tensor = torch.tensor(X_val.reshape(-1, 1, 48, 48), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val-1, dtype=torch.long)
    
    in_channels = 1  # grayscale images
    num_classes = 4
    batch_size = 32  # Adjust based on your classification task
    results_list = []
    # Combine training data into a PyTorch dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("beginning model calculations...")
    
    for num_conv_layers in range(2, 6):
    # Experiment with different kernel sizes
        for kernel_size in range(2, 8):  # trying kernel sizes from 3 to 7
        
        # Experiment with different epoch sizes
            for num_epochs in [20]:  # Adjusting the epoch sizes 
                try:
                    # Instantiate the model with dropout
                    model = FlexibleCNN(in_channels, num_classes, num_conv_layers=num_conv_layers, kernel_size=kernel_size, dropout_rate=0.5)
    
                    #  loss function and optimizer with weight decay
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adjust weight_decay as needed
    
                    # Training loop
                    for epoch in range(num_epochs):
                        model.train()  # Set the model to training mode
                        for inputs, labels in train_loader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
    
                    # Evaluate the model on the test set
                    model.eval()  # Set the model to evaluation mode
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    test_accuracy = correct / total
    
                    # Evaluate the model on the training set (optional)
                    model.eval()
                    correct_train = 0
                    total_train = 0

                    with torch.no_grad():
                        for inputs, labels in train_loader:
                            outputs = model(inputs)
                            _, predicted_train = torch.max(outputs, 1)
                            total_train += labels.size(0)
                            correct_train += (predicted_train == labels).sum().item()
                    train_accuracy = correct_train / total_train
                    print(f"Convolutional Layer Size: {num_conv_layers} Kernel Size: {kernel_size} Epoch Size: {num_epochs} Val Accuracy: {test_accuracy} Train Accuracy: {train_accuracy}")
                    # Append results to the DataFrame
                    results_list.append({
                        "Convolutional_Layer_Size": num_conv_layers,
                        "Kernel_Size": kernel_size,
                        "Epoch_Size": num_epochs,
                        "Val_Accuracy": test_accuracy,
                        "Train_Accuracy": train_accuracy
                    })
                except RuntimeError as e:
                    print(f"RuntimeError: {e}")

    results_df = pd.DataFrame(results_list)
# Print the DataFrame with all results
    print(results_df)
    results_df.to_csv("model_results.csv", index=False)
    
    results_df['Accuracy_Difference'] = abs(results_df['Train_Accuracy'] - results_df['Val_Accuracy']) / results_df['Val_Accuracy'] * 100

# Find the model with the smallest accuracy difference
    best_model = results_df.loc[results_df['Accuracy_Difference'].idxmin()]
    
    print("Best Model:")
    print(best_model)
    
    results_df.sort_values(by='Val_Accuracy', ascending=False)
    
    model1 = FlexibleCNN(1, 4, num_conv_layers=4, kernel_size=3, dropout_rate=0.5)
    criterion1 = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop for Model 1
    for epoch in range(20):
        model1.train()
        for inputs, labels in train_loader:
            optimizer1.zero_grad()
            outputs = model1(inputs)
            loss = criterion1(outputs, labels)
            loss.backward()
            optimizer1.step()

    with open("mainmodel.pkl", "wb") as file:
        pickle.dump(model1, file)
    print('Main Model Saved as - mainmodel.pkl')    
        
    #Model 2
    model2 = FlexibleCNN(1, 4, num_conv_layers=2, kernel_size=3, dropout_rate=0.5)
    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop for Model 2
    for epoch in range(20):
        model2.train()
        for inputs, labels in train_loader:
            optimizer2.zero_grad()
            outputs = model2(inputs)
            loss = criterion2(outputs, labels)
            loss.backward()
            optimizer2.step()    
            
    with open("variant1.pkl", "wb") as file:
       pickle.dump(model2, file)         
    print('Variant 1 Saved as - variant1.pkl')    
   
       
    #Model 3
    model3 = FlexibleCNN(1, 4, num_conv_layers=4, kernel_size=5, dropout_rate=0.5)
    criterion3 = nn.CrossEntropyLoss()
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop for Model 3
    for epoch in range(20):
        model3.train()
        for inputs, labels in train_loader:
            optimizer3.zero_grad()
            outputs = model3(inputs)
            loss = criterion3(outputs, labels)
            loss.backward()
            optimizer3.step()       
            
    with open("variant2.pkl", "wb") as file:
      pickle.dump(model3, file)
      print('Variant 2 Saved as - variant2.pkl')    

     
    with open("mainmodel.pkl", "rb") as file:
      loaded_model1 = pickle.load(file)

# Set the model to evaluation mode
    print('Running Main Model Evaluation')
    loaded_model1.eval()
    
    # Prepare test data
    X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 48, 48), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test the model and save actual vs predicted emotions
    correct = 0
    total = 0
    predictions_main = {"Actual": [], "Predicted": []}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = loaded_model1(inputs)
            _, predicted = torch.max(outputs, 1)
    
            # Append actual and predicted values to the dictionary
            predictions_main["Actual"].extend(labels.numpy())
            predictions_main["Predicted"].extend(predicted.numpy())
    
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate test accuracy
    test_accuracy = correct / total
    print("Test Accuracy Main Model:", test_accuracy)
    
    # Create a DataFrame from the dictionary
    predictions_main_df = pd.DataFrame(predictions_main)
    
    # Save the DataFrame to a CSV file
    predictions_main_df.to_csv("predictions_main.csv", index=False)
    
    print('Running Variant 1 Evaluation')

    with open("variant1.pkl", "rb") as file:
      loaded_model2 = pickle.load(file)

    #   Set the model to evaluation mode
    loaded_model2.eval()

# Prepare test data
    X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 48, 48), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test the model and save actual vs predicted emotions
    correct = 0
    total = 0
    predictions_variant1 = {"Actual": [], "Predicted": []}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = loaded_model2(inputs)
            _, predicted = torch.max(outputs, 1)
    
            # Append actual and predicted values to the dictionary
            predictions_variant1["Actual"].extend(labels.numpy())
            predictions_variant1["Predicted"].extend(predicted.numpy())
    
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate test accuracy
    test_accuracy = correct / total
    print("Test Accuracy variant 1:", test_accuracy)
    
    # Create a DataFrame from the dictionary
    predictions_variant1_df = pd.DataFrame(predictions_variant1)
    
    # Save the DataFrame to a CSV file
    predictions_variant1_df.to_csv("predictions_variant1.csv", index=False)

    print('Running Variant 2 Evaluation')

# Load the trained model
    with open("variant2.pkl", "rb") as file:
        loaded_model3 = pickle.load(file)
    
    # Set the model to evaluation mode
    loaded_model3.eval()
    
    # Prepare test data
    X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 48, 48), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test the model and save actual vs predicted emotions
    correct = 0
    total = 0
    predictions_variant2 = {"Actual": [], "Predicted": []}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = loaded_model3(inputs)
            _, predicted = torch.max(outputs, 1)
    
            # Append actual and predicted values to the dictionary
            predictions_variant2["Actual"].extend(labels.numpy())
            predictions_variant2["Predicted"].extend(predicted.numpy())
    
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate test accuracy
    test_accuracy = correct / total
    print("Test Accuracy Variant 2:", test_accuracy)
    
    # Create a DataFrame from the dictionary
    predictions_variant2_df = pd.DataFrame(predictions_variant2)
    
    # Save the DataFrame to a CSV file
    predictions_variant2_df.to_csv("predictions_variant2.csv", index=False)

# Initializing empty lists to store actual and predicted labels
    actual_labels = []
    predicted_labels = []

# Iterating through the test set
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
    
            # Append actual and predicted labels to the lists
            actual_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())
    
    # Create a DataFrame with actual and predicted labels
    actual_vs_predicted_df = pd.DataFrame({
        'Actual_Label': actual_labels,
        'Predicted_Label': predicted_labels
    })
    
    # Print the DataFrame
    print(actual_vs_predicted_df)
    
    # Save the DataFrame to a CSV file
    actual_vs_predicted_df.to_csv("actual_vs_predicted.csv", index=False)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get confusion matrices for each model
    conf_matrix_model1 = get_confusion_matrix(model1, test_loader)
    conf_matrix_model2 = get_confusion_matrix(model2, test_loader)
    conf_matrix_model3 = get_confusion_matrix(model3, test_loader)
    
    # Plot the confusion matrices using seaborn heatmap
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(conf_matrix_model1, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels='emotions', yticklabels='emotions')
    plt.title('Confusion Matrix - Main Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(conf_matrix_model2, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels='emotions', yticklabels='emotions')
    plt.title('Confusion Matrix - Variant 1')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(conf_matrix_model3, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels='emotions', yticklabels='emotions')
    plt.title('Confusion Matrix - Variant 2')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.show()
    
    actual_labels = predictions_main_df['Actual']  # Replace with the actual labels from your test data
    predicted_labels = predictions_main_df['Predicted']  # Replace with the predicted labels from your model1
    
    labels = actual_labels.unique()
    
    # Initialize a dictionary to store confusion matrices for each label
    confusion_matrices = {}
    
    # Calculate and store confusion matrix for each label
    for label in labels:
        cm = confusion_matrix(actual_labels == label, predicted_labels == label)
        confusion_matrices[label] = cm
    
    # Display or plot confusion matrices
    for label, cm in confusion_matrices.items():
    
        # Plot the confusion matrix
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix for Label {label}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    actual_labels_main = predictions_main_df["Actual"]
    predicted_labels_main = predictions_main_df["Predicted"]
    
    actual_labels_variant1 = predictions_variant1_df["Actual"]
    predicted_labels_variant1 = predictions_variant1_df["Predicted"]
    
    actual_labels_variant2 = predictions_variant2_df["Actual"]
    predicted_labels_variant2 = predictions_variant2_df["Predicted"]
    
    # Calculate metrics for each model
    metrics_main = calculate_metrics(actual_labels_main, predicted_labels_main)
    metrics_variant1 = calculate_metrics(actual_labels_variant1, predicted_labels_variant1)
    metrics_variant2 = calculate_metrics(actual_labels_variant2, predicted_labels_variant2)
    
    # Create a DataFrame to display the metrics
    metrics_df = pd.DataFrame({
        'Model': ['Main Model', 'Variant 1', 'Variant 2'],
        'Macro P': [round(metrics_main['Macro Precision'], 3), round(metrics_variant1['Macro Precision'], 3), round(metrics_variant2['Macro Precision'], 3)],
        'Macro R': [round(metrics_main['Macro Recall'], 3), round(metrics_variant1['Macro Recall'], 3), round(metrics_variant2['Macro Recall'], 3)],
        'Macro F': [round(metrics_main['Macro F1 Score'], 3), round(metrics_variant1['Macro F1 Score'], 3), round(metrics_variant2['Macro F1 Score'], 3)],
        'Micro P': [round(metrics_main['Micro Precision'], 3), round(metrics_variant1['Micro Precision'], 3), round(metrics_variant2['Micro Precision'], 3)],
        'Micro R': [round(metrics_main['Micro Recall'], 3), round(metrics_variant1['Micro Recall'], 3), round(metrics_variant2['Micro Recall'], 3)],
        'Micro F': [round(metrics_main['Micro F1 Score'], 3), round(metrics_variant1['Micro F1 Score'], 3), round(metrics_variant2['Micro F1 Score'], 3)],
        'Accuracy': [round(metrics_main['Accuracy'], 3), round(metrics_variant1['Accuracy'], 3), round(metrics_variant2['Accuracy'], 3)],
    })
    
    # Print the DataFrame
    print(metrics_df)
 
if __name__ == "__main__":
    main()    