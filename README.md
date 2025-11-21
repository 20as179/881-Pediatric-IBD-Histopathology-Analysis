# HistologyTrialModule
Custom 3D Slicer module that automates hematoxylin and eosin stain separation of histology images.
It is designed to improve the visualization of histology stain separation by providing simultaneous views of the original image, hematoxylin channel, and eosin channels in each of Slicer's viewers. It also contains a basic customized VGG16 binary classification model that provides a prediction for IBD as Crohn's disease (CD) or ulcerative colitis (UC).

## Installation
To download the module extension, download and extract the zip file HistologyTrialModule. Main code for execution is in HistologyTrialModule.py. The VGG16 model is located in the Resources folder as VGG16_binary_classification.keras and is called upon in the Pycharm file to run binary classification of input data.

## Usage
The module accepts a volumetric stack of histology images. To properly utilize the model, whole slide images (WSI)s are cropped and resized to the same size and stacked to create a 3D volumetric array of size = (depth, height, width) where depth = the number of slices/images in the stack. Preprocessing code for this step can be found in ______.ipynb. A sample volume (______.nrrd) has been provided and contains preprocessing of a full set of histology images from the Children’s Hospital of Orange County (CHOC) Research Institute of a pediatric patient with IBD. Upon loading into 3D Slicer, navigate to the HistologyTrialModule extension and click apply to transform the model into its separate hematoxylin and eosin staining channels.

## Training
### WSI Cropping and Resizing
Preprocessing is located in ______.ipynb. The first step focuses on cropping the raw images (.jpgs) into 1000x1000 pixel fields such that outer borders of whitespace (noise) are removed. Images must be of the same size in order to be stacked, and dimensions less than 1000 pixels are filled in with empty values. The volumes are saved as .nrrd files and stored under each patient's directory as OUTPUT_VOLUME.
### VGG16 Model Training
Training is located in ________.ipynb. VGG16 accepts inputs of 224x224, so each slice of an input is iteratively downsized to match requirements. Data augmentation makes further adjustments to the images using the following parameters in ImageDataGenerator:
- insert
- parameters
- adjust
- later
For customized data augmentation, these parameters can be adjusted according to the user's wishes for diverse results.

The model was pretrained using ImageNet weights and its layers have been customized based on other histology classification models using VGG16 as a baseline. The customized layers are adjusted to account for binary classification of histology images. The model uses weakly supervised training of pediatric IBD data from 23 patients (951 individual slides in total) with each slide classified as UC or CD depending on the patient's overall diagnosis. A sample csv file is provided at ________.csv containing each slide's information and label from the training subset of the data.

In 3D Slicer, the output for "Diagnosis" will either be UC or CD based on the average of predictive scores for the full stack of slices.
