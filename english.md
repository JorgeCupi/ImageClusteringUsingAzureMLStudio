# Image clustering using Python and Azure Machine Learning Studio #
These are the steps that we will follow to accomplish our mission:
- Understanding what image clustering is and what we need to do to achieve it
- Selecting a dataset
- Image preprocessing with Python
- Clustering with Azure ML Studio
- Conclusion and results

## Understanding what image clustering is and what we need to do to achieve it ##
So yes, we want to cluster images but why and how should be our main questions before any coding begins.
### Why? ###
Image clustering could be useful when we have a large dataset of images and we want to identify patterns within them, there are different examples:
- Can I cluster these car images into different car brands or colors?
- Can I cluster these x-ray photos to identify potential anomalies?
- Can I cluster a set of people faces to identify emotions on their expressions?
- Can I cluster these flowers to identify different species or different growth states between a same flower?

As you can see the opportunies are endless and all we'll need to do is to preprocess such images before applying AI into them. 

>NOTE. For simplicity purposes we'll use the terms AI, Machine Learning (ML) and Deep Learning as if they were the same altough they are not. But for our current task here we won't need to settle down and understand their differences.

In our current repo we'll try to answer this question: *Can a Machine Learning model grab a flower picture and determine if the flower is a rose or a dandelion?*

Spoiler alert? Yes, it can.
### How? ###
Now that we know our purpose it's time to determine how we will do it. We need to pre process our images in such a way that our ML model is able to diggest it to first be trained and then start to 'classify' new images arriving.

>NOTE. We'll cover some basic Machine Learning topics so if you're a medium-advanced user you might jump straight to Image preprocessing with Python and/or Clustering with Azure ML Studio

### New to data science? ###
We'll try to run a quick Data Science 101 here:
- Any data science project needs a business question to be answered
- Once we have it, we need to look for a big dataset available to answer such question
- We'll split the dataset into a training and a test set, sometimes we'll even have a validation set.
- Why split the data? Because we'll use ~80% of it to train a machine learning model (training set) and the remaining 20% to test the trained model with unseen data and see if its accurate and precise
- Once the data is split we'll use it to train multiple ML algorithms and check the results to see which algorithm works best of us
- Depending on the results obtained there's a chance we'll need to use different algorithms or do some parameter tuning on the ones that got the best performance
- Finally, we'll draw conclusions from our results and determine if we are happy with what we'be obtained or we'll need more data, use different algorithms, have a different approach, etc.

### Understanding labels and features ###
Most Machine Learning models take datasets that contain a set of features and a single label per every data entry. Then, for each entry the model grabs those features and makes calculations to try to predict the original label.

So, we could think of things like:
- Number of petals
- Color of flower
- Size of plant
- Number of leafs
- Diameter of flower, etc

as features, and our label would be the already identified type of flower:

<table>
<tr>
<td>Type</td>
<td># of leafs</td>
<td># of petals</td>
<td>Color</td>
</tr>
<tr>
<td>Rose</td>
<td>6</td>
<td>28</td>
<td>Red</td>
</tr>
<tr>
<td>Dandelion</td>
<td>4</td>
<td>6</td>
<td>Yellow</td>
</tr>
<tr>
<td>Rose</td>
<td>5</td>
<td>24</td>
<td>White</td>
</tr>
<tr>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<td>Rose</td>
<td>7</td>
<td>32</td>
<td>Red</td>
</table>

**Our first column in the table above is our label and the rest are features**

### Supervised vs unsupervised learning ###
When we have a label to start the we are talking about a classification problem and thus we'll use 'supervised learning' algorithms such as Decision Trees, Two-class Bayes Point Machine, and others.
When we don't have (or don't want to use) an initial label then we'll need to use 'unsupervised learning' algorithms, K-Means clustering being one of the most used.
#### Why would we ignore the label? ####
In some cases we might want to validate our dataset and for that we'll use clustering algorithms first. If the clustering model gives us a same number of labels then that *should* mean our labels are good.

### Converting an image to a set of features and labels ###
If ML models take labels and features, then we'll need to convert our image into that. What type of approach could we take?
- We could look at each image and create a csv file containing color, number of leafs, color of the flower, calculate the diameter of the flower and turn this into a classification problem
- Or we could convert the image into an array of features where each feature is a pixel of the image and face a clustering problem

Let's take the second approach.

Imagine you have a 2px x 2px color image:

![Image: 2x2 matrix representing an image](https://codesciencelove.files.wordpress.com/2013/06/img-thing.jpg)

If each of those squares above represent a pixel then we could represent our image as a two dimension array in python:
```python
image = [ 
[pixel, pixel],
[pixel, pixel]
]
```
But let's remember that a pixel has 3 numeric values that represent their red, green and blue values (RGB), so we'll have a three dimension array:
```python
image = [
[ [1,2,3], [4,5,6] ],
[ [7,8,9], [0,1,2] ]
]
```
So, a 2px x 2px image has in fact 12 numeric features representing each color of every pixel. We could use those 12 features and train a clustering model with them and see what clusters we'll get in return.

## Selecting the dataset ##
We'll use the [Flower color images](https://www.kaggle.com/olgabelitskaya/flower-color-images) dataset from [Olga Belitskaya](https://www.kaggle.com/olgabelitskaya) found in [Kaggle.com](https://www.kaggle.com/).

>TIP: If you are new in Data Science, Kaggle.com is THE place to go for finding datasets, experiments and challenges from one of the biggest data science communities in the web. 

### Analizing the dataset ###
There are 210 images and all of them have 128px x 128px, so that means that per each image we'll have 49152 feautures (remember the RGB values of each pixel)

#### Is it important that the images have the same pixel size? ####
It is! Remeber that each pixel with its 3 color values will represent a feauture for our algorithms. If we had a 128x128 image and then another one of 128x156 pixels the difference between them in terms of features would be huge:
- First image = 128x128x3 = 49152
- Second image = 128x156x3 = 59904 features

Think of it, there's a difference of 10752 features between them. If we would write a csv file with features as columns per each image that would mean the 128x128 image will have 10752 blank features and when our models are training those blank features would make a whole difference. Even if the image is the exact same but on a different resolution.

#### What could be done standardize the image sizes? ####
The first step would be to ensure our dataset have similar, if not same, sizes for every image. If this isn't possible because the images were taken with different cameras, on different zooming levels, etc then we could write some scripts to change the size of all the images to a standard size

## Image preprocessing with Python ##
### Change size of images ###
Thankfully, the images in our dataset have the same size, but if you need to change sizes to your own dataset here' a quick script in python for this task:

```python
import numpy as np
from PIL import Image
import os

rootdirs = ['./trainingSet/','./testSet/']
resizedFolder = ['./resizedTrainingSet/','./resizedTestSet/']
i=0
j=0
k=0
myDirs = []
myResizedFolders = []
folderNames = []
for x in rootdirs:
    for subdir, dirs, files in os.walk(x):
        if len(dirs) >0:    
            for folder in dirs:
                os.makedirs(resizedFolders[k]+folder)
                myResizedFolders.append(resizedFolders[k]+folder)
                folderNames.append(folder)
        else:
            for file in files:
                    im = Image.open(x+folderNames[j]+'/'+file)
                    imResize = im.resize((128,128), Image.ANTIALIAS)
                    imResize.save(myResizedFolders[j]+'/'+ str(i)+'.jpg', 'JPEG', quality=90)
                    i=i+1
            j=j+1
    k = k+1
    j=0
    myResizedFolders = []
    folderNames = []
```
### Understanding the resizing script ###
The script above asumes you already have 2 folders called **trainingSet** and **testSet** with your data split in a training and a test set. Once the script runs it will create two new folders called **resizedTrainingSet** and **resizedTestSet**.

Per every image found in the original training and test folders the script will open an image, change the size to 128 x 128 pixels and save it to its respective training or test new resized folder. 

**The code can be found in the 'code' folder at the root directory of this repo.**

 
## Clustering with Azure ML Studio ##

## Conclusion and results ##