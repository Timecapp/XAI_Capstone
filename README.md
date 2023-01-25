# XAI_Capstone
The repo for FourthBrain.AI Capstone Industry Sponsored: "Explainability in Healthcare" from Samsung, to explore methods of breast cancer detection in whole slide histopathology images, with explainability.

Team Members:  Dr. Shaista Hussain & Mr. Pedram Emami
Cohort:  MLE-10, 2022-2023



Introduction
Diagnosing breast cancer requires a gold standard histopathological confirmation; conducted by a pathologist who microscopically visualizes dissected tissue. The analysis of the specimen results in a diagnosis and stage if positive, with a justification of observed findings.  AI models trained on object detection/segmentation tasks are currently used as a diagnostic tool, as they provide precise and consistent predictions. However, the explanations from these models, such as heatmaps, similar images and high level features, do not match those of an experienced pathologist. Explainable features that are suitable for histopathological image diagnosis need to be defined. The data annotations and choice of models needs to be designed to generate accurate diagnostic predictions with high quality explainable features. The annotation and explanation process involved could be more precise and consistent with accurate diagnostic algorithms that identify specific and measurable morphological abnormalities that reflect a carcinoma based on objective annotations.

Key Research Questions
We aim to address the following:
Replicate state of the art explainable features (heat maps, similar images in training data, high level image features) for object detection/segmentation based prediction models, and identify suitability and limitations to histopathological images for cancer diagnosis.
Implement other well known, model agnostic explainable features such as LIME and SHAP with histopathological images and determine suitability & limitations, to improve explainability on previous models, and satisfy physician and patient end-users.
Does increasing a training data set size and any other factors in dataset/annotations improve explainability  in this case?
(optional) Explore use of additional text annotations from pathologists, characterising the malignant patches to generate textual descriptions for predictions. 
(optional) Explore use of anomaly detection and/or graph neural networks instead of object detection/segmentation to improve accuracy of diagnosis and explainability.
Perform detailed literature survey for the state-of-the-art techniques for explainability in histopathological images for breast cancer. 

Datasets
1. The Wisconsin-Breast Cancer (Diagnostics) dataset (WBC) was downloaded from the UCI machine learning repository.
It is a classification dataset, which records the measurements for breast cancer cases.
There are two classes, benign and malignant.
The malignant class of this dataset is downsampled to 21 points, which are considered as outliers, while points in the benign class are considered inliers.
The dataset was created by Dr. William H. Wolberg, at the University Of Wisconsin Hospital at Madison, Wisconsin,USA.
Dr. Wolberg used fine needle aspirates from patients with solid breast masses and digitally analysed cytological features.
The program uses a curve-fitting algorithm, to compute ten features from each one of the cells in the sample, than it calculates the mean value, extreme value and standard error of each feature for the image, returning a 30 real-valuated vector.
DISTRIBUTION

Train 70%
Test 30%
FORMAT

Attribute Information:
1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

10 Features computed for each cell nucleus:
a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)

2. SOURCE Breast Cancer Histopathological Image Classification (BreakHis) Dataset

BreakHis dataset was downloaded from the Laboratory Robotic Vision & Imaging.

composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X).

contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format).

dataset is divided into two main groups: benign tumors and malignant tumors. - benign refers to a tissue architecture that does not match malignancy criteria (e.g., increased cellular atypia, mitosis, disruption of basement membranes, metastasize, etc).

malignant lesion is locally destructive, invasive and metastasizes.
DISTRIBUTION Train 80% Test 20%

FORMAT

image filenames store information about: method of biopsy procedure, tumor class, tumor type, patient identification, and magnification factor.

eg. SOB_B_TA-14-4659-40-001.png is image 1, with magnification factor 40X, of a benign tumor of type tubular adenoma, original from slide 14-4659, which was collected by procedure SOB. (Spanhol et al., 2017)

Format of image file name is given by the following BNF notation:

---- ::=UNDER ::=M|B ::=< BENIGN_TYPE>| ::=A|F|PT|TA ::=DC|LC|MC|PC ::= ::= ::= ::=40|100|200|400 ::=| ::= ::| ::=0|1|…|9 ::=A|B|…|Z

Methodology
- In our model, the images (Whole Slide Image or WSI) are digitally produced by "Whole Slide Scanners" which are then to be annotated independantly by histopathologists to mark the Regions of Interest. If there is a difference between annotated information amongst the histopathologists, the slide is then send to another expert for review.

Machine learning transformations could include colourings and hues in the image datasets, segmentation and others.
The WSI is then sliced into multiple small segments to expedite processing times.
The slices are marked as 'Blank' or NoN-IDC, or as a Region of Interest.
Blanks are discarded; Non-IDC or ROI slices are then fed into the models and a final model is then used to predict the validation data.
The aggregated model is also for xAI (explainability).

- WBC Models & Baseline Comparison

CNN / Keras https://www.kaggle.com/code/luckyapollo/predict-idc-in-breast-cancer-histology-images/edit

Huang, Y (2020). Wisconsin Dataset . https://towardsdatascience.com/explainable-deep-learning-in-breast-cancer-prediction-ae36c638d2a4 used LIME

to explain the IDC image prediction results of a 2D ConvNet model in breast cancer diagnosis.
Explanations of model prediction of both IDC and non-IDC were provided by setting the number of super-pixels/features (i.e., the num_features parameter in the method get_image_and_mask()) to 20.
observed that the explanation results are sensitive to the choice of the number of super pixels/features.
explanations via highlighted bounding boxes
recommendations:
Domain knowledge to adjust parameters
Quality of the input data (pixels)
Accuracy can be improved by adding more samples.
Mooney, IDC Breast Cancer. https://www.kaggle.com/code/paultimothymooney/predicting-idc-in-breast-cancer-histology-images/notebook
76% accuracy
recommendations:
Improve data visualization
Optimize data augmentation
Optimize NN architecture

Workflow
- 

Results

Discussion

Conclusions
