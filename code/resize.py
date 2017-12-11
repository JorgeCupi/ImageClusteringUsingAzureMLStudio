import numpy as np
from PIL import Image
import os

rootdirs = ['./trainingSet/','./testSet/']
resizedFolder = ['./resizedTrainingSet/','./resizedTestSet/']
os.makedirs(resizedFolder[0])
os.makedirs(resizedFolder[1])
i=0
j=0
myDirs = []
for x in rootdirs:
    for subdir, dirs, files in os.walk(x):
        for file in files:
            i=i+1
            im = Image.open(x+file)
            imResize = im.resize((128,128), Image.ANTIALIAS)
            imResize.save(resizedFolder[j]+ str(i)+'.png', 'PNG', quality=90)
        j=j+1
    