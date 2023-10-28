# Project README

## File Structure

### 1. Resources

#### - icml_face_data.csv link_on_github/report/dataset_info.txt

- The dataset is used to obtain expression data for neutral and angry classes.

#### - Engaged_Images

  - Bored Images Consisting of:
    - bored_cropped
    - drowsy_cropped
    - looking_away_cropped

### 2. data_processing.py

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
   !conda install -c pytorch pytorch
   !conda install -c pytorch torchvision
   !conda install tqdm
   !conda install future
   !conda install matplotlib
   !conda install seaborn
   ```

   Please note that you may need to restart Spyder or the IPython console after installing packages for them to take effect.

4. To successfully run the code, extract the zip file.
5. Download and add the `icml_face_data.csv` file to the resources folder from the link provided in the report.
6. Run the code in Spyder after installing the required libraries.

## Notes

- The dataset 2 is linked in the report, but additional processing of images from that dataset was needed needed. So we are directly attaching those cropped images in directories in resources folder
