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