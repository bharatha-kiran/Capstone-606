# Capstone-606
Capstone Project on Bone cancer detection using Image Segmentation
Group:

Madhu Kiran, Bharatha
Sai Ketan, Gande
Charan Reddy, Jakkireddy

Project Title: Bone Cancer Detection Using Image Segmentation

Abstract:
Bone cancer is a relatively uncommon type of cancer, but it is a serious disease which becomes dangerous if not diagnosed in the early stages. Generally, bone cancer diagnosis involves imaging tests and if the surgeon finds something unusual, the patient is subjected to more advanced tests. Our project is collecting the raw images and training the processed data set with pretrained models (bottleneck, traditional CNN) comparing the accuracy and then we have segmented images leveraging Apeer(online tool) which has been fed into the U-net model and comparing all the models to find out which model performs best for our data.

Introduction:
Image segmentation is a crucial step in medical image processing for the identification of bone cancer. In medical images of the bones, such as X-ray or MRI images, the objective is to automatically recognize and segment bone tumors or other abnormalities. Patients with bone cancer may benefit from this in terms of diagnosis and therapeutic preparation.

Using deep learning models, such as Convolutional Neural Networks (CNNs), for picture segmentation is a typical method for detecting bone cancer (Khobragade, 2022). CNNs, which are neural networks created especially for image processing tasks, have demonstrated success in applications for medical image analysis.

Given its high accuracy in identifying cancerous tissues from medical images, the results of the cancer detection model using MobileNet can be encouraging. However, several variables, including the caliber of the input image data, the preprocessing methods employed, and the hyperparameters of the MobileNet model, affect the model's performance.

Early cancer detection can benefit from using picture data and MobileNet for cancer detection. The approach can help clinicians diagnose patients more correctly and have better patient outcomes. To improve the model's effectiveness and validate its application in clinical settings, additional study is necessary. One type of CNN architecture that has been used for bone cancer detection is the U-Net architecture. The U-Net architecture is a fully convolutional neural network that was originally designed for biomedical image segmentation. It is called U-Net because of its U-shaped architecture, which consists of a contracting path and an expanding path. The contracting path captures the context, and the expanding path enables precise localization.

Methods:

Data collection: The first step is to collect the data required for Osteosarcoma cancer detection. This may involve collecting medical images, such as X-rays or MRI scans, of patients with and without Osteosarcoma cancer.

Data cleaning: Since the data is from different sets, we have combined all the images into one folder and performed normalization and resized the image before training them.

Image preprocessing: We have divided the entire image dataset into train and test folders dividing the images equally across two folders.
we use train and test data generators in machine learning to evaluate the performance of our model and prevent overfitting. The data generators allow us to efficiently train and test our models on large datasets, while data augmentation techniques can help to increase the variety of the data and improve the performance of the model.

Labeling: We have labelled the data using the below 4 classes.
(Viable, non-viable, non-tumor, viable: non-viable)

Data splitting: The labeled images are then split into training, validation, and testing sets. This ensures that the machine learning model is trained on a subset of the data and tested on a separate subset to evaluate its performance.

Sequence of steps after data preprocessing:

1.	Used a ML classification model (SVC) because we have a csv file with features and labels got an accuracy of 45%. The model has the below scores.
Accuracy: 0.415
Precision: 0.415
Recall: 0.282
F1-score: 1.000





2.	Used a basic sequential CNN model and improved accuracy to 70%.
 
3.	Used transfer learning, A pretrained CNN model called Mobile Net and got a slightly improved accuracy but according to our results we can see overfitting for our data hence thought of segmenting our images and try different approach.
   

4.	We have used online tool Apeer to create semantic segmentation and created masks that can be inputted along with original images into a pretrained model called UNET which is good for medical imaging and try to get the best accuracy.

Conclusion:

Each model's accuracy, precision, recall, and F1 score can be assessed using the data from the training and testing phases. These metrics offer a numerical evaluation of the model's effectiveness in identifying malignant tissues in medical images. Comparing the performance of CNN, MobileNet, and UNet can also reveal important details about which model, given the particulars of the dataset and the issue at hand, is more useful for cancer diagnosis. In general, CNNs have been utilized extensively in the analysis of medical pictures and have demonstrated promising outcomes in the detection of cancer from medical images. However, MobileNet is a lightweight deep neural network that may not be as accurate as CNNs or UNet for medical image analysis, making it less suitable for mobile applications. For image segmentation tasks, U-Net is a well-liked model that might be helpful in pinpointing the precise position and extent of malignant tissues. With the use of CNN, Mobile Net, and U-Net training on cancer picture data, we expect our models to conclude that these models are effective in detecting cancer. The dataset, the problem, the available computational resources, and the desired level of accuracy are only a few of the variables that influence the optimal model to use.

REFERENCES:

(Khobragade, 2022). Revaluating Pretraining in Small Size Training Sample Regime. 
https://www.academia.edu/88103650/Revaluating_Pretraining_in_Small_Size_Training_Sample_Regime

Bone Cancer Detection from X-Ray and MRI Images through Image Segmentation Techniques
https://www.ijrte.org/wp-content/uploads/papers/v8i6/F7159038620.pdf

Bone Cancer Detection Using Feature Extraction Based Machine Learning Model
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8712164/

Deep Learning for Classification of Bone Lesions on Routine MRI
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8190437/

AX-U-net: A Deep Learning Framework for Image Segmentation to Assist Pancreatic Tumor Diagnosis
https://www.frontiersin.org/articles/10.3389/fonc.2022.894970/full

Git-hub Reference:

Multiclass semantic segmentation using U-Net 
https://github.com/bnsreenu/python_for_microscopists/blob/master/208-simple_multi_unet_model.py

Insight-face-erase
https://github.com/admshumar/insight-face-erase
