
# Challenge: Pneumonia detector

Given a lungs x-ray dataset, we created a model that can classify the lung as "Normal" or "Pneumonia", avec un score de 90%+. Moreover, we tried to make the lightest possible model: 2500 parameters, without any convolution.

### 1. Data analysis

We started from the [chest x-ray image](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/) from kaggle.

#### Problem: Unequal data distribution
The dataset is rather badly balanced: 1300 images of "Normal" lungs and 3875 of "Pneumonia" lungs. We decided, during the data transformation, to reduce the size our dataset in order to have a more balanced distribution of the data.

#### Problem: Multiple sizes images


### 2. Data transformation

### 3. Model evaluation