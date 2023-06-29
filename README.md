# Chest X-ray Abnormalities Classification
This project aims to develop a robust classification system for analyzing chest X-ray images and identifying 14 distinct types of thoracic abnormalities present in chest radiographs. The objective is to assist medical professionals in making accurate diagnoses by automating the process of abnormality detection and classification.

### Installation
```
pip install -r requirements.txt
```
### Usage
Navigate to the project directory:

```
cd chextnet    
```

### Dataset
#### VinBigData Chest-Xray Abnormalities Detection: 
The dataset comprises 18,000 postero-anterior (PA) CXR scans in DICOM format, which were de-identified to protect patient privacy.
These annotations were collected via VinBigData's web-based platform, VinLab. Details on building the dataset can be found in our recent paper “VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations”.
All images were labeled by a panel of experienced radiologists for the presence of 14 critical radiographic findings as listed below:

0 - Aortic enlargement

1 - Atelectasis

2 - Calcification

3 - Cardiomegaly

4 - Consolidation

5 - ILD

6 - Infiltration

7 - Lung Opacity

8 - Nodule/Mass

9 - Other lesion

10 - Pleural effusion

11 - Pleural thickening

12 - Pneumothorax

13 - Pulmonary fibrosis

This figure below illustrates the data for Exploratory Data Analysis (EDA) concerning 14 different types of thoracic abnormalities. In this data, a label of 0 denotes a normal case, while a label of 1 signifies the presence of abnormalities

![eda.png](..%2F..%2F..%2FDownloads%2Feda.png)

### Proposed Method 
![img.png](docs%2Fimgs%2Fimg.png)

### Model Training

#### 1. Segment lungs
```
bash segment-lung.sh    
```

#### 2. Preprocess the dataset
```
bash scripts/prepare-dataset.sh    
```

#### 3. Train the dataset
```
bash scripts/train-classifier.sh    
```

### Evaluation and Result
![img_1.png](img_1.png)

### References
Z. Chen, X. Wei, P. Wang and Y. Guo, "Multi-label image recognition with graph convolutional networks", Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 5177-5186, 2019.

Rajpurkar, P. et al. CheXNet: radiologist-level pneumonia detection on chest X-rays with deep learning. Preprint at https://arxiv.org/abs/1711.05225 (2017).

