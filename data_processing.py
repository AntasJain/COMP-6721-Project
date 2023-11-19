from __future__ import print_function
import os
import pandas as pd
import numpy as np
import torch
import torchvision
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random
from sklearn.model_selection import train_test_split
import csv

#Class with helper functions to generate image data in the desired format
class Generate_data():


    def str_to_image(self, str_img = ' '):
        '''
        Convert string pixels from the csv file into np image array
        '''
        imgarray_str = str_img.split(' ')
        imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
        return Image.fromarray(imgarray)

    def save_images(self, csv_file_name):
        '''
        save_images is a function responsible for saving images from data files e.g(train, test) in a desired folder
        '''
        folder_name = os.path.splitext(csv_file_name)[0]  # Extract the folder name from the CSV file name

        csvfile_path = os.path.join('resources', csv_file_name)
        folder_path = os.path.join('resources', folder_name)
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        data = pd.read_csv(csvfile_path)
        images = data[' pixels']
        number_of_images = images.shape[0]

        for index in tqdm(range(number_of_images)):
            img = self.str_to_image(images[index])
            img.save(os.path.join(folder_path, '{}{}.jpg'.format(folder_name, index)), 'JPEG')

        print('Done saving {} data'.format(folder_name))
        
    def read_image_data_as_array(self,file_path):
        """
        Convert .jpg image files to np image array
        """
        image_files = [file for file in os.listdir(file_path) if file.endswith(".jpg")]
        output_directory = csvfile_path = os.path.join('resources', "test_cropped_4848")
        if not os.path.exists(output_directory):
               os.mkdir(output_directory)
        images = []
        for image_file in image_files:
                output_path = os.path.join(output_directory, image_file)
                path = os.path.join(file_path, image_file)
                image = Image.open(path)
                image_transformed = image.convert("L")
                image_transformed = image_transformed.resize((48,48))
                #image_transformed.save(output_path)
                image_array = np.array(image_transformed, dtype='uint8').flatten()
                images.append(image_array)
                
        return images
    
        
    
        
    def class_distribution_diagram(self,shuffled_full_data):
        plt.figure(figsize=(8, 6))
        sns.countplot(x='emotion', data=shuffled_full_data)
        plt.title('Class Distribution in the Dataset')
        plt.xlabel('Emotion Class')
        plt.ylabel('Count')
        plt.show()
   
       
    def apply_transform(self,image_array):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=random.uniform(0.3, 0.7), contrast=random.uniform(0.3, 0.7)),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.ToTensor(),
        ])
        image = Image.fromarray(image_array[0], 'L')
        augmented_image = transform(image)
        augmented_array = np.array(augmented_image)
        return augmented_array.reshape(augmented_array.shape + (1,)) if len(augmented_array.shape) == 2 else augmented_array

    def pixel_distribution(self,shuffled_df):
        image_files = []

        for pixels, emotion in zip(shuffled_df['pixels'].head(25), shuffled_df['emotion'].head(25)):  # Select the first 25 samples
            # Ensure that pixel values are of type uint8
            image = Image.fromarray(np.array(pixels, dtype=np.uint8).reshape(48, 48))
            image_files.append((np.array(image), emotion))

        # Create a 5x5 grid
        fig, axes = plt.subplots(5, 5, figsize=(48, 48))

        # Tally the counts of each emotion
        emotion_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for i, (ax, (image, emotion)) in enumerate(zip(axes.ravel(), image_files)):
            ax.set_aspect('equal')
            ax.imshow(image, cmap='gray')
            ax.set_title(f'Emotion: {emotion}')
            ax.axis('off')

            # Update emotion counts
            emotion_counts[emotion] += 1

        plt.tight_layout()
        plt.show()

        # Display emotion counts
        print("Emotion Counts in the Grid:")
        for emotion, count in emotion_counts.items():
            print(f"Emotion {emotion}: {count}")

        # Plot histogram of pixel intensities
        plt.figure(figsize=(10, 6))
        for pixels, emotion in zip(shuffled_df['pixels'].head(25), shuffled_df['emotion'].head(25)):
            pixel_values = np.array(pixels, dtype=np.uint8).flatten()
            plt.hist(pixel_values, bins=256, range=[0, 256], alpha=0.7, label=f'Emotion {emotion}')

        plt.title('Pixel Intensity Distribution for Random Images')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

#Calculating pixel densities
def pixel_density(row):
    total_pixels = row.size

    # Calculate the number of non-zero pixels (or non-blank pixels)
    non_zero_pixels = np.count_nonzero(row)

    # Calculate pixel density as a ratio
    pixel_density = non_zero_pixels / total_pixels
    return pixel_density
#use as a measure for brightness of the image
def calculate_intensity(row):
    return row.mean()


def view_image(data):
    
    # Filter rows where pixel density is less than or equal to 0.90
    data_to_plot = data

    # Create a 5x5 grid
    fig, axes = plt.subplots(10, 10, figsize=(48, 48))

    # Iterate through the filtered DataFrame and plot images
    for i, (ax, row) in enumerate(zip(axes.ravel(), data_to_plot.iterrows())):
        _, data = row
        pixels = np.array(data['pixels'], dtype=np.uint8).reshape(48, 48)
        emotion = data['emotion']

        ax.set_aspect('equal')
        ax.imshow(pixels, cmap='gray')  # greyscale images
        ax.set_title(f'Emotion: {emotion}\nPixel Density: {data["pixel_density"]:.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Function to enhance brightness and contrast of an image
def enhance_brightness_contrast(image, brightness_factor=1.5, contrast_factor=0.5):
    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Enhance brightness using ImageEnhance
    brightness = ImageEnhance.Brightness(pil_image)
    brightness_enhanced_image = brightness.enhance(brightness_factor)

    # Enhance contrast using ImageEnhance
    contrast = ImageEnhance.Contrast(brightness_enhanced_image)
    contrast_enhanced_image = contrast.enhance(contrast_factor)

    # Convert back to numpy array
    enhanced_pixels = np.array(contrast_enhanced_image)

    return np.clip(enhanced_pixels, 0, 255).astype(np.uint8)



def main():
        # Set a target number of samples
        target_count = 750

        # Create a list to store augmented DataFrames
        augmented_dataframes = []
        image_files = []

        generate_data = Generate_data() 
        """
        Loading the data

        The ICML face data is available as a .csv which contains the neutral and angry emotion clases and is loaded into a 
        dataframe.
        The engaged and neutral classes are available as .jpg images which are first converted into image arrays and then loaded in
        dataframes.
        These dataframes are then merged to get the full_data dataframe containing all the 4 emotion classes
        """
        #To get the data for neutral and angry expression
        data_icml_face_data_path = os.path.join('resources', 'icml_face_data.csv')
        data_icml_face_data = pd.read_csv(data_icml_face_data_path)
        
        #To get the engaged emotion data
        generate_data = Generate_data()
        data_engaged = generate_data.read_image_data_as_array(os.path.join('resources', 'engaged_images'))
        
        generate_data = Generate_data()
        data_bored = generate_data.read_image_data_as_array(os.path.join('resources', 'bored_cropped'))
        
        generate_data = Generate_data()
        data_engaged_clicked = generate_data.read_image_data_as_array(os.path.join('resources', 'engaged_self_created'))
        
        
        generate_data = Generate_data()
        data_drowsy = generate_data.read_image_data_as_array(os.path.join('resources', 'drowsy_cropped'))
        
        generate_data = Generate_data()
        data_looking_away = generate_data.read_image_data_as_array(os.path.join('resources', 'looking_away_cropped'))
        
        print("Length of neutral data: ", len(data_icml_face_data[data_icml_face_data["emotion"]==6]))
        total_data_engaged = data_engaged+data_engaged_clicked
        print("Length of engaged data : ", len(total_data_engaged))
        print("Length of bored data : ", len(data_bored)," Length of drowsy data : ", len(data_drowsy)," Length of looking away data : ", len(data_looking_away))
        data_bored = data_bored + data_drowsy + data_looking_away
        print("Length of combined bored data : ", len(data_bored))
        print("Length of angry data : ", len(data_icml_face_data[data_icml_face_data["emotion"]==0]))
        #converting string value of pixels to image array and changing type to unit8
        data_icml_face_data['pixels'] = data_icml_face_data[' pixels'].apply(lambda string_image: np.array(string_image.split(), dtype='uint8'))

        #selecting only required data from ICML face data
        df_neutral_angry = data_icml_face_data[data_icml_face_data["emotion"].isin([0,6])]

        #dropping unrequired columns
        drop_columns = [' Usage',' pixels']
        df_neutral_angry = df_neutral_angry.drop(columns=drop_columns)

        #changing emotion class labels
        df_neutral_angry["emotion"][df_neutral_angry["emotion"]==0] = 4
        df_neutral_angry["emotion"][df_neutral_angry["emotion"]==6] = 1
        df_neutral_angry = df_neutral_angry[['pixels','emotion']]
        print(df_neutral_angry.head())

        #create dataframe for engaged, emotion value = 2
        df_engaged = pd.DataFrame()
        df_engaged["pixels"] = total_data_engaged
        df_engaged["emotion"] = 2
        print(df_engaged.head())

        #create dataframe for bored, emotion value = 3
        df_bored = pd.DataFrame()
        df_bored["pixels"] = data_bored
        df_bored["emotion"] = 3
        print(df_bored.head())

        #merging the dataframes
        full_data = pd.concat([df_neutral_angry, df_engaged, df_bored], ignore_index = True)
        print(full_data.head())

        print(full_data.info())

        print(full_data['emotion'].value_counts())

        #Shuffle the data
        shuffled_full_data = full_data.sample(frac=1, random_state=42)  
        # Reset the index to make it continuous
        shuffled_full_data = shuffled_full_data.reset_index(drop=True)
        print("Shuffled DataFrame:")
        print(shuffled_full_data.head())

        map_emotions_to_str={
            1:"neutral",
            2:"engaged",
            3:"bored",
            4:"angry"
        }
        shuffled_full_data["emotion_name"] = shuffled_full_data["emotion"].map(map_emotions_to_str)
        print(shuffled_full_data.head())


        generate_data.class_distribution_diagram(shuffled_full_data)


        # Iterate over unique emotion categories
        for emotion in full_data['emotion'].unique():
            class_subset = full_data[full_data['emotion'] == emotion]
            
            # Check if the count is less than the target_count
            if len(class_subset) < target_count:
                # Calculate the remaining count needed for augmentation
                remaining_count = target_count - len(class_subset)

                # Define a data augmentation transformation
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=random.uniform(0.3, 0.7), contrast=random.uniform(0.3, 0.7)),
                    transforms.RandomResizedCrop(48, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                    transforms.ToTensor(),
                ])

                # Apply data augmentation to reach the target_count
                augmented_samples = [
                    {'pixels': (transform(torch.tensor(np.expand_dims(row['pixels'].reshape((48, 48)), axis=0), dtype=torch.uint8)).squeeze().numpy().flatten() * 255).astype(int), 'emotion': emotion}
                    for _, row in class_subset.sample(remaining_count, replace=True).iterrows()
                ]

                # Combine original and augmented samples for the category
                augmented_class_df = pd.concat([class_subset, pd.DataFrame(augmented_samples)])

                # Append the augmented class DataFrame to the list
                augmented_dataframes.append(augmented_class_df)
            else:
                # If the count is greater than or equal to the target_count, sample 1000 without replacement
                sampled_subset = class_subset.sample(target_count, replace=False)
                augmented_dataframes.append(sampled_subset)

        # Combine augmented class DataFrames into one
        augmented_data_df = pd.concat(augmented_dataframes)

        # Save the augmented data to a new CSV file
        augmented_data_df.to_csv("augmented_data.csv", index=False)

        print(augmented_data_df.head())

        #check if the data is correctly represented in the image array
        length_of_image_arrays = full_data['pixels'].str.len()
        min_length = length_of_image_arrays.min()
        max_length = length_of_image_arrays.max()
        print("Min Len:", min_length,"Max Len:", max_length)

        generate_data.class_distribution_diagram(augmented_data_df)

        print([augmented_data_df['emotion'].unique()])

        # Shuffle the DataFrame to get a random order of samples
        shuffled_df = augmented_data_df.sample(frac=1).reset_index(drop=True)
        generate_data.pixel_distribution(shuffled_df)

        augmented_data_df['pixel_density'] = augmented_data_df['pixels'].apply(pixel_density)
        augmented_data_df.head()
        print(augmented_data_df['pixel_density'].quantile([0,0.25,0.50,0.75,0.90,0.99]))
        print(augmented_data_df['pixel_density'].quantile([0,0.02,0.05,0.20,0.25]))
        print(len(augmented_data_df[augmented_data_df['pixel_density']<=0.894523]))


        # Filter rows where pixel density is less than or equal to 0.90
        low_quality_images_df = augmented_data_df[augmented_data_df['pixel_density'] <= 0.894523]

        # Create a 5x5 grid
        fig, axes = plt.subplots(10, 10, figsize=(48, 48))

        # Iterate through the filtered DataFrame and plot images
        for i, (ax, row) in enumerate(zip(axes.ravel(), low_quality_images_df.iterrows())):
            _, data = row
            pixels = np.array(data['pixels'], dtype=np.uint8).reshape(48, 48)
            emotion = data['emotion']

            ax.set_aspect('equal')
            ax.imshow(pixels, cmap='gray')  #greyscale images
            ax.set_title(f'Emotion: {emotion}\nPixel Density: {data["pixel_density"]:.2f}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        low_quality_indices = low_quality_images_df.index
        cleaned_augmented_data_df = augmented_data_df.drop(index=low_quality_indices)

        generate_data.class_distribution_diagram(cleaned_augmented_data_df)

        #remove the low pixel density 
        condition = augmented_data_df['pixel_density'] <= 0.894523
        augmented_data_df = augmented_data_df[~condition]

        plt.figure(figsize=(10, 6))

        # Sample random 100 images from the DataFrame
        random_sample_df = cleaned_augmented_data_df.sample(100, random_state=random.randint(1,100))

        for pixels, emotion in zip(random_sample_df['pixels'], random_sample_df['emotion']):
            pixel_values = np.array(pixels).flatten()
            plt.hist(pixel_values, bins=256, range=[0, 256], alpha=0.7, label=f'Emotion {emotion}')

        plt.title('Pixel Intensity Distribution for Random 100 Images')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

        print(cleaned_augmented_data_df.head())

        cleaned_augmented_data_df['intensity'] = cleaned_augmented_data_df['pixels'].apply(calculate_intensity)
        print(cleaned_augmented_data_df['intensity'].quantile([0,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]))
        #plot the images which have brightness below 100 to check the brightness
        #brighten the image and change contrast to see affect   
        low_val = cleaned_augmented_data_df['intensity'].quantile([0.10])
        print(low_val[0.10])


        view_image(cleaned_augmented_data_df[cleaned_augmented_data_df["intensity"] <= low_val[0.10]])
        print(len(cleaned_augmented_data_df[cleaned_augmented_data_df["intensity"] <= low_val[0.10]]))



        # Filter rows where intensity is less than 50
        low_intensity_df = cleaned_augmented_data_df[cleaned_augmented_data_df["intensity"] <= low_val[0.10]]

        # Randomly select 10 images from the filtered DataFrame
        random_low_intensity_samples = low_intensity_df.sample(10, random_state=random.randint(1, 100))

        # Create a 2x10 grid for original and enhanced images
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))

        # Iterate through the selected samples and plot original and enhanced images
        for i, (ax, (_, data)) in enumerate(zip(axes.T, random_low_intensity_samples.iterrows())):
            pixels = np.array(data['pixels'], dtype=np.uint8).reshape(48, 48)
            emotion = data['emotion']

            # Plot original image
            ax[0].set_aspect('equal')
            ax[0].imshow(pixels, cmap='gray')
            ax[0].set_title(f'Emotion: {emotion}\nOriginal')
            ax[0].axis('off')

            # Enhance brightness and contrast
            enhanced_pixels = enhance_brightness_contrast(pixels, brightness_factor=1.5, contrast_factor=0.5)

            # Plot enhanced image
            ax[1].set_aspect('equal')
            ax[1].imshow(enhanced_pixels, cmap='gray')
            ax[1].set_title(f'Emotion: {emotion}\nEnhanced')
            ax[1].axis('off')

        plt.tight_layout()
        plt.show()

        hi_val = cleaned_augmented_data_df['intensity'].quantile([0.99])
        view_image(cleaned_augmented_data_df[cleaned_augmented_data_df["intensity"] >= hi_val[0.99]])

        filtered_df = cleaned_augmented_data_df[cleaned_augmented_data_df['intensity'] < hi_val[0.99]]

        # Display the images with high intensity

        # Show the DataFrame after filtering
        print("Number of rows before filtering:", len(cleaned_augmented_data_df))
        print("Number of rows after filtering:", len(filtered_df))

        plt.figure(figsize=(10, 6))

        # Sample random 100 images from the DataFrame
        random_sample_df_1 = filtered_df.sample(100, random_state=random.randint(1,100))

        for pixels, emotion in zip(random_sample_df_1['pixels'], random_sample_df_1['emotion']):
            pixel_values = np.array(pixels).flatten()
            plt.hist(pixel_values, bins=256, range=[0, 256], alpha=0.7, label=f'Emotion {emotion}')

        plt.title('Pixel Intensity Distribution for Random 100 Images')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

        generate_data.class_distribution_diagram(filtered_df)
        np.set_printoptions(threshold=np.inf)
        
        length_of_image_arrays = filtered_df['pixels'].str.len()
        min_length = length_of_image_arrays.min()
        max_length = length_of_image_arrays.max()
        print("Min Len Filtered:", min_length,"Max Len Filtered:", max_length)
        filtered_df['pixels'] = filtered_df['pixels'].apply(lambda x: ','.join(map(str, x)))
        filtered_df.to_csv("finalized_data.csv", index=False)

        
        print('Execution Complete')
if __name__ == "__main__":
    main()