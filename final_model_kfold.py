from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


best_val_accuracy = 0.0
early_stopping_counter = 3
patience = 3
early_stopping_threshold = 0.001
# Define the FlexibleCNN class
class FlexibleCNN(nn.Module):
    # ... (Class definition remains the same)
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

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
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

# Part 2: K-fold Cross-Validation
def k_fold_cross_validation(X, y, k=10):
    global best_val_accuracy, early_stopping_counter  # Declare global variables
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results_list = []  # Move this outside the loop
    outer_metrics_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('Fold', ''), ('Macro', 'Precision'), ('Macro', 'Recall'), ('Macro', 'F1'), ('Micro', 'Precision'), ('Micro', 'Recall'), ('Micro', 'F1'), ('Accuracy', '')]))
    for num_conv_layers in [4]:
        # Experiment with different kernel sizes
        for kernel_size in [5]:
            # Reset variables for each model
            print(f"\nModel: Conv Layers={num_conv_layers}, Kernel Size={kernel_size}")
            print(f"Highest Accuracy Till Now: {best_val_accuracy}")
            
            # Variables to store results for each model
            model_results = {"Convolutional_Layer_Size": [], "Kernel_Size": [], "Fold": [], "Val_Accuracy": []}
            best_val_accuracy = 0.0
            early_stopping_counter = 0
            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                best_val_accuracy = 0.0
                early_stopping_counter = 0
                print(f"\nFold {fold + 1}/{k}:")

                X_train, X_val = X[train_index], X[test_index]
                y_train, y_val = y[train_index], y[test_index]

                # Convert data to PyTorch tensors
                X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 48, 48), dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long)

                X_val_tensor = torch.tensor(X_val.reshape(-1, 1, 48, 48), dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val - 1, dtype=torch.long)

                in_channels = 1  # grayscale images
                num_classes = 4
                batch_size = 32  # Adjust based on your classification task
                num_epochs = 20

                # Combine training data into a PyTorch dataset
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                test_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                print("beginning model calculations...")

                try:
                    # Instantiate the model with dropout
                    model = FlexibleCNN(in_channels, num_classes, num_conv_layers=num_conv_layers,
                                        kernel_size=kernel_size, dropout_rate=0.5)

                    # loss function and optimizer with weight decay
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

                        # Evaluate the model on the validation set
                        model.eval()
                        correct = 0
                        total = 0

                        with torch.no_grad():
                            for inputs, labels in test_loader:
                                outputs = model(inputs)
                                _, predicted = torch.max(outputs, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                            val_accuracy = correct / total
                            print(f"Epoch {epoch + 1}/{num_epochs} Val Accuracy: {val_accuracy}")
                            # Check for improvement based on the threshold
                            if val_accuracy > best_val_accuracy + early_stopping_threshold:
                                best_val_accuracy = val_accuracy
                                early_stopping_counter = 0
                            else:
                                early_stopping_counter += 1
                                print(f"Early stopping counter: {early_stopping_counter}/{patience}")

                            if early_stopping_counter >= patience:
                                print(f"Early stopping at epoch {epoch + 1}")
                                break
                            
                        # Print metrics for each epoch (optional)
                        

                        # Store results for each fold
                        model_results["Convolutional_Layer_Size"].append(num_conv_layers)
                        model_results["Kernel_Size"].append(kernel_size)
                        model_results["Fold"].append(fold + 1)
                        model_results["Val_Accuracy"].append(val_accuracy)
                        model.eval()  # Set the model to evaluation mode

                        # Iterate through the test loader to get predictions
                        all_predictions = []
                        all_actuals = []

                        with torch.no_grad():
                            for inputs, labels in test_loader:
                                outputs = model(inputs)
                                _, predicted = torch.max(outputs, 1)
                                all_predictions.extend(predicted.tolist())
                                all_actuals.extend(labels.tolist())

                        
                    # After training for each fold, append results to the DataFrame
                        conf_matrix_model1 = confusion_matrix(all_actuals, all_predictions)


                        # Calculate metrics for Model 1
                    metrics_model1 = calculate_metrics(all_actuals, all_predictions)

                        # Create a DataFrame to display the metrics
                    metrics_df_model1 = pd.DataFrame({
                        'Model': ['Model 1'],
                        ('Macro', 'Precision'): [round(metrics_model1['Macro Precision'], 3)],
                        ('Macro', 'Recall'): [round(metrics_model1['Macro Recall'], 3)],
                        ('Macro', 'F1'): [round(metrics_model1['Macro F1 Score'], 3)],
                        ('Micro', 'Precision'): [round(metrics_model1['Micro Precision'], 3)],
                        ('Micro', 'Recall'): [round(metrics_model1['Micro Recall'], 3)],
                        ('Micro', 'F1'): [round(metrics_model1['Micro F1 Score'], 3)],
                        ('Accuracy', ''): [round(metrics_model1['Accuracy'], 3)],
                        })


                      
                    outer_metrics_df = pd.concat([outer_metrics_df, metrics_df_model1], ignore_index=True,axis=0) 
                    results_list.append(model_results)
                except RuntimeError as e:
                    print(f"RuntimeError: {e}")

    # Combine results for all folds into a single DataFrame
    results_df = pd.concat([pd.DataFrame(model_results) for model_results in results_list], ignore_index=True)

    average_results = results_df.groupby(['Convolutional_Layer_Size', 'Kernel_Size']).agg({
        'Val_Accuracy': 'mean'
    }).reset_index()
   
    # Print highest accuracy till now
    print(f"\nHighest Accuracy Till Now: {best_val_accuracy}")
    print(outer_metrics_df) 
    
    # ... (Part 2 code)

# Part 3: Final Model Building and Evaluation


def main():
    # Part 1: Data Preprocessing and Splitting
    final_df =pd.read_csv(r"sampled_dataset.csv")    
    final_df['pixels'] = final_df['pixels'].apply(lambda x: np.fromstring(x, sep=',', dtype=int))
    X = np.array(final_df['pixels'].tolist())
    y = np.array(final_df['emotion'])  # Assuming labels start from 1
    k_fold_cross_validation(X, y, k=10)
    #final_df = pd.read_csv(r"C:\Users\14389\Documents\AI\Python project\Python project\data_classified_gender_age_full2.csv")
    #final_df['pixels'] = final_df['pixels'].apply(lambda x: np.fromstring(x, sep=',', dtype=int)) # Convert string back to list or array
    length_of_image_arrays = final_df['pixels'].str.len()
    min_length = length_of_image_arrays.min()
    max_length = length_of_image_arrays.max()
    print(max_length,min_length)
    train_data, val_data, test_data = split_test(final_df)

    print(final_df.info())
    
    X = np.array(final_df['pixels'].tolist())
    y = np.array(final_df['emotion'])
    age = final_df['age']
    gender=final_df['gender']

    X_train, X_test_val, y_train, y_test_val, age_train, age_test_val, gender_train, gender_test_val = train_test_split(X, y, age, gender, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test, age_val, age_test, gender_val, gender_test = train_test_split(X_test_val, y_test_val, age_test_val, gender_test_val, test_size=0.5, random_state=42)

    # Print lengths
    print("Length of X_train:", len(X_train[0]))
    print("Length of y_train:", len(y_train))
    print("Length of age_train:", len(age_train))
    print("Length of X_val:", len(X_val))
    print("Length of y_val:", len(y_val))
    print("Length of age_val:", len(age_val))
    print("Length of X_test:", len(X_test[0]))
    print("Length of y_test:", len(y_test))
    print("Length of age_test:", len(age_test))


    
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
    early_stopping_threshold = 0.001
    results_list = []
    # Combine training data into a PyTorch dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("beginning model calculations...")
    
    for num_conv_layers in [4]:
    # Experiment with different kernel sizes
        for kernel_size in [5]:  # trying kernel sizes from 3 to 7
        #scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, threshold=0.01, verbose=True)
        # Experiment with different epoch sizes
            for num_epochs in [20]:  # Adjusting the epoch sizes 
                try:
                    best_val_accuracy = 0.0
                    early_stopping_counter = 0
                    patience = 3
                    #print(f"\nFold {fold + 1}/{k}:")

                    in_channels = 1  # grayscale images
                    num_classes = 4
                    batch_size = 32  # Adjust based on your classification task
                    num_epochs = 20

                    # Combine training data into a PyTorch dataset
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    test_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    print("beginning model calculations...")

                    try:
                        # Instantiate the model with dropout
                        model = FlexibleCNN(in_channels, num_classes, num_conv_layers=num_conv_layers,
                                            kernel_size=kernel_size, dropout_rate=0.5)

                        # loss function and optimizer with weight decay
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

                            # Evaluate the model on the validation set
                            model.eval()
                            correct = 0
                            total = 0

                            with torch.no_grad():
                                for inputs, labels in test_loader:
                                    outputs = model(inputs)
                                    _, predicted = torch.max(outputs, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                                val_accuracy = correct / total
                                print(f"Epoch {epoch + 1}/{num_epochs} Val Accuracy: {val_accuracy}")
                                # Check for improvement based on the threshold
                                if val_accuracy > best_val_accuracy + early_stopping_threshold:
                                    best_val_accuracy = val_accuracy
                                    early_stopping_counter = 0
                                else:
                                    early_stopping_counter += 1
                                    print(f"Early stopping counter: {early_stopping_counter}/{patience}")

                                if early_stopping_counter >= patience:
                                    print(f"Early stopping at epoch {epoch + 1}")
                                    break

                            # Print metrics for each epoch (optional)
                    except Exception as e:
                            print(e)

                except Exception as e:
                            print(e)
        X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 48, 48), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test-1, dtype=torch.long)
        test__final_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_final_loader = DataLoader(test__final_dataset, batch_size=batch_size, shuffle=False)
        # After training loop, predict on the test set
        model.eval()
        y_true_test = []
        y_pred_test = []

        with torch.no_grad():
            for inputs, labels in test_final_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())
        # Create a DataFrame with the required information for test data
        print(f"After model{len(X_test[0])}")
        df_results = pd.DataFrame({
            'pixels': [pixel_array.flatten() for pixel_array in X_test],
            'actual': np.array(y_true_test),
            'predicted': np.array(y_pred_test),
            'age': age_test,
            'gender':gender_test# Use the correct age array for the test set
        })
        #print(f"After df{len(df_results['pixels'][0])}")
        df_results['pixels'] = df_results['pixels'].apply(lambda x: ' '.join(map(str, x)))
        # Save the DataFrame to a CSV file
        # Save the DataFrame to a CSV file
        df_results.to_csv(r"model_results_age.csv", index=False)
        save_model(model, r"model_final_before_bias_mitigation.pth")
        
    df = df_results    
    age_groups = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    
    # Group by age
    grouped_df = df.groupby('age')
    
    # Loop through age groups
    for age, group in grouped_df:
    
        # Create a binary classification for each age group
        actual = group['actual']
        predicted = group['predicted']  # Assuming 'predicted' column contains the model's predictions
    
        # Calculate metrics
        precision = precision_score(actual, predicted, average='macro')  # Use 'macro' for multiple classes
        recall = recall_score(actual, predicted, average='macro')  # Use 'macro' for multiple classes
        f1 = f1_score(actual, predicted, average='macro')  # Use 'macro' for multiple classes
        accuracy = accuracy_score(actual, predicted)
    
        # Store metrics and age information
        age_groups.append(age)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
    
    # Create a new DataFrame for the results
    results_df_age = pd.DataFrame({
        'Age': age_groups,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1 Score': f1_scores,
        'Accuracy': accuracy_scores
    })
    
    # Display the results
    print(results_df_age)

        ############################################## MAIN #####################################################
    gender_groups = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    
    # Group by age
    grouped_df = df.groupby('gender')
    
    # Loop through age groups
    for gender, group in grouped_df:
    
        # Create a binary classification for each age group
        actual = group['actual']
        predicted = group['predicted']  # Assuming 'predicted' column contains the model's predictions
    
        # Calculate metrics
        precision = precision_score(actual, predicted, average='macro')  # Use 'macro' for multiple classes
        recall = recall_score(actual, predicted, average='macro')  # Use 'macro' for multiple classes
        f1 = f1_score(actual, predicted, average='macro')  # Use 'macro' for multiple classes
        accuracy = accuracy_score(actual, predicted)
    
        # Store metrics and age information
        gender_groups.append(gender)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
    
    # Create a new DataFrame for the results
    results_df_gender = pd.DataFrame({
        'Gender': gender_groups,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1 Score': f1_scores,
        'Accuracy': accuracy_scores
    })
    
    # Display the results
    print(results_df_gender)
    
    class_labels = [0, 1, 2, 3]
    
    # Define class names corresponding to the labels
    class_names = {0: 'Neutral', 1: 'Engaged', 2: 'Bored', 3: 'Angry'}
    
    # Extract true and predicted labels
    true_labels = df['actual']
    predicted_labels = df['predicted']
    
    # Create a confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[class_names[label] for label in class_labels],
                yticklabels=[class_names[label] for label in class_labels])
    plt.title('Confusion Matrix For All Classes')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
        
    
if __name__ == "__main__":
    main()
