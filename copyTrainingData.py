import os
from shutil import copyfile
import random

for file in os.listdir('./TrainingData'):
    if not file.endswith('.jpg'):
        continue
    n = random.uniform(0, 1)
    if (n > 0.8):
        copyfile('./TrainingData/'+file, r"C:\Users\User\Desktop\Court Athena\Text-detection\darknet\build\darknet\x64\data\0-9/"+file)
        fileName = os.path.splitext(file)[0]
        copyfile('./TrainingData/'+fileName+'.txt', r"C:\Users\User\Desktop\Court Athena\Text-detection\darknet\build\darknet\x64\data\0-9/"+fileName+'.txt')