import os
from shutil import copyfile
import random

for file in os.listdir('./TrainingData'):
    if not file.endswith('.jpg'):
        continue
    targetFolder = r"C:\Users\User\Desktop\Court Athena\Text-detection\darknet\build\darknet\x64\data\digits/"
    n = random.uniform(0, 1)
    if (n > 0.8):
        copyfile('./TrainingData/'+file, targetFolder+file)
        fileName = os.path.splitext(file)[0]
        copyfile('./TrainingData/'+fileName+'.txt', targetFolder+fileName+'.txt')