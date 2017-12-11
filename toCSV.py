import numpy as np
import csv
from PIL import Image
import os
import csv

rootdir = ['./trainingSet/','./testSet/']
csvFiles = ['trainingSet.csv','testSet.csv']

labels = []
fileTraining = open('trainingLabels.csv')
reader = csv.reader(fileTraining)
trainingLabels = list(reader)
labels.append(trainingLabels)
fileTest = open('testLabels.csv')
reader = csv.reader(fileTest)
testLabels = list(reader)
labels.append(testLabels)

j=0
k=0
finalArray = []
for directory in rootdir:
    currentLabels = labels[k]
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            fileName = subdir+file
            myImage = np.array(Image.open(fileName))
            myArray = myImage.flatten()
            imageArray = myArray.tolist()
            imageArray[:0]=[currentLabels[j]]
            finalArray.append(imageArray)
            j = j + 1
        print(directory+ " images flat proccess completed")

        with open(csvFiles[k], "wb") as f: 
            writer = csv.writer(f)
            writer.writerows(finalArray)
        print(csvFiles[k]+ " written successfully")
        finalArray = []
        j = 0
        k = k + 1