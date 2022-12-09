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

Methodology

Workflow

Results

Discussion

Conclusions
