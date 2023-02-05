
# XAI_Capstone
The repo for FourthBrain.AI Capstone Industry Sponsored: "Explainability in Healthcare" from Samsung, to explore methods of breast cancer detection in whole slide histopathology images, with explainability.

Team Members:  Dr. Shaista Hussain & Mr. Pedram Emami
Cohort:  MLE-10, 2022-2023

Presentation link: https://docs.google.com/presentation/d/1rI4byBdGlJn_1nwsFZLaZarMxek-hsHrtHivRO4l0WU/edit?usp=sharing

# Introduction
Diagnosing breast cancer requires a gold standard histopathological confirmation; conducted by a pathologist who microscopically visualizes dissected tissue. The analysis of the specimen results in a diagnosis and stage if positive, with a justification of observed findings.  AI models trained on object detection/segmentation tasks are currently used as a diagnostic tool, as they provide precise and consistent predictions. However, the explanations from these models, such as heatmaps, similar images and high level features, do not match those of an experienced pathologist. Explainable features that are suitable for histopathological image diagnosis need to be defined. The data annotations and choice of models needs to be designed to generate accurate diagnostic predictions with high quality explainable features. The annotation and explanation process involved could be more precise and consistent with accurate diagnostic algorithms that identify specific and measurable morphological abnormalities that reflect a carcinoma based on objective annotations.

# Key Research Questions
We aim to address the following:
Replicate state of the art explainable features (heat maps, similar images in training data, high level image features) for object detection/segmentation based prediction models, and identify suitability and limitations to histopathological images for cancer diagnosis.
Implement other well known, model agnostic explainable features such as LIME and SHAP with histopathological images and determine suitability & limitations, to improve explainability on previous models, and satisfy physician and patient end-users.
Does increasing a training data set size and any other factors in dataset/annotations improve explainability  in this case?
(optional) Explore use of additional text annotations from pathologists, characterising the malignant patches to generate textual descriptions for predictions. 
(optional) Explore use of anomaly detection and/or graph neural networks instead of object detection/segmentation to improve accuracy of diagnosis and explainability.
Perform detailed literature survey for the state-of-the-art techniques for explainability in histopathological images for breast cancer. 

# Datasets
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

# FORMAT

image filenames store information about: method of biopsy procedure, tumor class, tumor type, patient identification, and magnification factor.

eg. SOB_B_TA-14-4659-40-001.png is image 1, with magnification factor 40X, of a benign tumor of type tubular adenoma, original from slide 14-4659, which was collected by procedure SOB. (Spanhol et al., 2017)

Format of image file name is given by the following BNF notation:

---- ::=UNDER ::=M|B ::=< BENIGN_TYPE>| ::=A|F|PT|TA ::=DC|LC|MC|PC ::= ::= ::= ::=40|100|200|400 ::=| ::= ::| ::=0|1|…|9 ::=A|B|…|Z

# Methodology
- In our model, the images (Whole Slide Image or WSI) are digitally produced by "Whole Slide Scanners" which are then to be annotated independantly by histopathologists to mark the Regions of Interest. If there is a difference between annotated information amongst the histopathologists, the slide is then send to another expert for review.

Machine learning transformations could include colourings and hues in the image datasets, segmentation and others.
The WSI is then sliced into multiple small segments to expedite processing times.
The slices are marked as 'Blank' or NoN-IDC, or as a Region of Interest.
Blanks are discarded; Non-IDC or ROI slices are then fed into the models and a final model is then used to predict the validation data.
The aggregated model is also for xAI (explainability).

# WBC Models & Baseline Comparison

CNN / Keras https://www.kaggle.com/code/luckyapollo/predict-idc-in-breast-cancer-histology-images/edit

# Huang, Y (2020). Wisconsin Dataset . https://towardsdatascience.com/explainable-deep-learning-in-breast-cancer-prediction-ae36c638d2a4 used LIME

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

# Workflow
We started with the baseline for Wisconsin Breast Cancer DS: https://colab.research.google.com/drive/1HkO0OURdTIXPkscuydexVyDWOZ2Vz3yJ?usp=sharing

Then the BreakHis DS baseline:
https://colab.research.google.com/drive/1jouMB5NGG4zvkKGn-i-RTkXlPLmm7Eek?usp=sharing

Then Wisconsin DS models: https://colab.research.google.com/drive/1s5x2IFtxgqB6jk98w3r5T3Hlntewf-Zm?usp=sharing
Then BreakHis DS models: https://colab.research.google.com/drive/1t0NEbD4E_SgV94kNCRSML1V-trH_frS7?usp=sharing


# Results
We began the dataset exploration with the Samsung issued Canine Breast Histopathology dataset, which we realized did work on an image feature selection basis, but only complicated the black box issue as there required a translation of histopathological features between canine to human species.  We proceeded to establish a dataset source - which we did 2 of; the first with the Wisconsin Breast Cancer Dataset (csv) and the second with the BreakHis dataset (whole slide images).  We conducted EDA on both and then proceeded to run a baseline based on Yu Huang’s use of LIME to explain the image prediction results of a 2D Convolutional Neural Network (ConvNet) for the Invasive Ductal Carcinoma (Huang, 2020).  Once the explanation of the model prediction was obtained, we applied get_image_and_mask() to get the template image and the corresponding mask image (super pixels) showing the boundary of the area of the IDC image in yellow.  Then we applied a LIME image explainer to explain the IDC image prediction results of the 2D Convoluted Network model of Invasive Ductal Carcinoma of the breast from digital pathology slides.  Having completed this task and achieving 86.9% accuracy on image features and fulfilled LIME explainability, similar to the author’s results, we proceeded to run K fold logistic regression, and SVM on the BreakHis dataset and achieved a similar 86% accuracy in detection.  In our EDA we explored univariate, bi variate and multivariate analyses as well as used heat maps and correlation matrices to establish what we were getting right and wrong.  We then tried several models on the Wisconsin Breast Cancer Dataset and after EDA and baseline, we tried a Logistic Regression model with an accuracy of 97.6%,  Naive Bayes at 94%, and SVM much higher at 98.2%, and Random FOREST AT 94.7%, where the  Decision Tree training set accuracy was 0.99 and test set accuracy was 94.7%.   
Then we tried to address the data imbalance, so we oversampled the minority class using the Synthetic Minority Oversampling Technique (SMOTE), and using Standard Scaler, and Log transformed skewed data, removed outliers with z-score, we achieved Random Forest accuracy is 96.3%, and successfully applied LIME, SHAP MOrris sensitivity partial sensitivity plots and gaussian ROC curves (0.99 AUC).


# Deployment
We are deploying via AWS 
