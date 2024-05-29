# Forest-Fire-Detection
Developing a CNN-based system for early forest fire detection is crucial to safeguard the environment, wildlife, and human lives. This project focuses on using deep learning techniques to create a model that can efficiently identify forest fires, enabling prompt and effective firefighting measures.
"Introducing our cutting-edge CNN-based Forest Fire Detection Model! Safeguarding the environment, wildlife, and lives is our priority. Early detection is key, and our advanced deep learning technology ensures swift identification of potential forest fires. Experience the power of innovation for effective and timely firefighting. Protect what matters most with our state-of-the-art solution!

## Data Collection
A total of 16036 images were gathered from the following online sources like Pixbay, Getty Images, Pexels, iStockphoto and Kaggle \
It’s worth noting that the collected images varied in size. Subsequently, these images have been
stored in an unstructured data format within Google Drive.

## Exploratory Data Analysis (EDA)
To assess the uniformity in the dimensions of images 
~~~
import cv2
import os
image_dir = r"C:\Users\Aditi\Desktop\ADS PROJECT\datacollection"
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    print(f"Image: {filename}, Height: {height}, Width: {width}")

~~~

## Data Pre-Processing
### Data resizing
All the images are resized into a size of 256*256

### Data Labeling
Classified into two distinct labels: Forest Fire Images and Non-Forest Fire Images.

### Data Balancing
Implemented an oversampling strategy by incorporating new images of forests,
appropriately resized. 

## Data Splitting
70-15-15 split for training, testing, and validation sets,

## Model Selection & Development
Choosing a Convolutional Neural Network (CNN) for forest fire detection is rooted in its capacity
to automatically learn hierarchical features from images. \
~~~
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import cv2
import matplotlib.pyplot as plt
~~~


## Output
Total accuracy obtained is approximately 0.916 i.e 91.6%
