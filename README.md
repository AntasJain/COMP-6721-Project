
# Project Title

A brief description of what this project does and who it's for

# Project README

## File Structure

### 1. Resources

#### - icml_face_data.csv
- [Link to Dataset on Google Drive](https://github.com/AntasJain/COMP-6721-Project](https://drive.google.com/file/d/1JFEbt_P0muFsT6chghMxcA87_2tImHc3/view?usp=share_link))
- Due to high data size, it cannot be completely uploaded to the github

The dataset is used to obtain expression data for neutral and angry classes.

#### - Engaged_Images
  - Bored Images Consisting of:
    - bored_cropped
    - drowsy_cropped
    - looking_away_cropped
#### - Curated Self images

From the Github Obtain the new file called Sampled_data.csv (for part 3), put it in the folder with the python files (not in resources).

### 2. data_processing.py

Contains code to run data preprocessing, augmentation and visualization, make sure to place all the python files where resources folder is located.

### 3. model_and_eval.py

Contains code to create, evaluate, and save models - mainmodel.pkl, variant1.pkl, variant2.pkl

### 4. test_on_image.py

Contains code to test any image against the main model.
Open Code and Change the path to the image in the code to successfully run the model.
Output will display image with label as predicted label.

### 5. testoncsv.py

Contains code to test on whole dataset, in this case we have by default put test.csv to be predicted.
A csv file "mainmodel_predictions" will be created when you run the file with actual and predicted labels column.

### 6. final_model_kfold.py

Contains code for Final Model Training Using Early Stopping Techniques, K-Fold Cross Validation, and code for evaluation on Gender and Age  Bias and Mitigation

## How to run Complete project
* Put all 5 python files in same folder as resources folder, run data_processing.py, as it finishes execution a curated, augmented and filtered dataset will be saved in folder,
* Run model_and_eval.py - this will save the most efficient model as 'mainmodel.pkl' and 2 variants as 'variant1.pkl' and 'variant2.pkl'. This will also create confusion matrix and evaluation tables. test, train and val csvs will be saved to same folder.
* To test an image, open the test_on_csv.py, put path to the image in variable and run code, output will be label predicted.
* To test complete dataset, put test.csv or train.csv or finalized_data.csv in the variable and run the code, a csv file will be saved with predicted vs actual label and also accuracy will be displayed on the console.
* To run k-fold and bias evaluation, put sampled_data.csv in the same folder, and run the file, all the analysis and final model selection will be executed.

## Necessary Libraries/Packages

- **os:** Provides a portable way to use operating system-dependent functionality.
- **pandas:** A popular data manipulation and analysis library.
- **numpy:** A library for numerical operations and working with arrays.
- **torch:** PyTorch, a deep learning framework for machine learning.
- **torchvision:** A package consisting of popular datasets, model architectures, and common image transformations for computer vision.
- **from torch.utils.data:** Specific modules and classes for working with data in PyTorch.
- **from torchvision import transforms:** Modules for image transformations in PyTorch.
- **from __future__ import print_function:** A Python 2 to Python 3 migration helper for enabling print function.
- **PIL:** Python Imaging Library, which is used for opening, manipulating, and saving many different image file formats.
- **tqdm:** A library for displaying progress bars in the console.
- **random:** To get random values.
## How to Run

1. Open Spyder.
2. In the top menu, click on "Tools" and then select "Open an IPython console."
3. In the IPython console, use conda commands to install the packages. For example:

  ```python
!conda install pandas numpy
!conda install -c pytorch pytorch torchvision
!conda install tqdm
!conda install future
!conda install matplotlib
!conda install seaborn
!conda install pickle
!conda install scikit-learn
