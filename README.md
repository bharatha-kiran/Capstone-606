# Bone Cancer detection using Image segmentation

### Group:
- Madhu Kiran, Bharatha
- Sai Ketan, Gande
- Charan Reddy, Jakkireddy

### Abstract
Bone cancer, particularly osteosarcoma, is a serious health concern requiring early and accurate detection. Deep learning models offer promising results in medical image analysis tasks. This project compares the effectiveness of various models for bone cancer detection: SVC, CNN, MobileNetV2, and U-Net. Raw image data obtained from various medical imaging techniques is utilized. Preprocessing techniques, including resizing, normalization, and augmentation, enhance the dataset. The dataset is divided into training and testing sets for evaluation

### Introduction
Osteosarcoma is the most common form of bone cancer, mainly affecting children and young adults. It is a malignant tumor that develops in the cells responsible for bone formation, known as osteoblasts. Early detection is vital for improving the prognosis and survival rates of individuals with osteosarcoma. Medical imaging techniques, such as X-rays, CT scans, and MRI, are commonly employed for diagnosis and monitoring the progression of the disease.
Image segmentation is a computer vision technique that involves dividing an image into distinct regions or objects to extract meaningful information. In the case of osteosarcoma, image segmentation can aid in identifying and outlining tumor regions within medical images, facilitating accurate diagnosis and treatment planning. The utilization of image segmentation in bone cancer detection offers several advantages. It enables precise measurement of tumor size, assessment of tumor characteristics, and monitoring changes over time. Additionally, it provides valuable insights into the tumor's location and its relationship with surrounding tissues.


### Data Set Description
-	Data Collection: The data was collected by a team of clinical scientists at the University of Texas Southwestern Medical Center, Dallas.
-	Patient Selection: Archival samples from 50 patients treated at Children's Medical Center, Dallas, between 1995 and 2015, were used to create this dataset.
-	Labeling: The images are labeled as Non-Tumor, Viable Tumor, and non-viable.
-	Dataset Size: The dataset consists of 1144 images of size 1024 X 1024 at 10X resolution.
- 	Class Distribution: The dataset comprises 536 (47%) non-tumor images, 263 (23%) non-viable images, and 345 (30%) viable tumor images.



### CONCLUSION:
In conclusion, CNN as expected has performed well in the binary classification. For multi-class classification the results would have been better with more data. The U-Net model has demonstrated good performance for the segmentation task based on the masks provided to it. The U-Net model's advanced architecture, incorporating the concept of skip connections and encoder-decoder structures, has allowed it to excel in tasks such as image segmentation and medical imaging. Its ability to capture fine-grained details and accurately identify boundaries and features has contributed to its good performance. 

#### Dataset: https://github.com/bharatha-kiran/Capstone-606/tree/main/capstoneimages%2Cmasks/edu

#### Code: https://github.com/bharatha-kiran/Capstone-606/blob/main/CAPSTONE.ipynb
	
#### Detailed Report: https://github.com/bharatha-kiran/Capstone-606/blob/main/PROJECT-REPORT-capstone.docx


