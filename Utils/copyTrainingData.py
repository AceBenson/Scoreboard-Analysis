import os
import shutil
import random

targetFolder = r"C:\Users\User\Desktop\Court Athena\Text-detection\darknet\build\darknet\x64\data\0-9/"
# print("Deleting files...")
# for filename in os.listdir(targetFolder):
#     file_path = os.path.join(targetFolder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))

print("Copying files...")
for file in os.listdir('./TrainingData'):
    if not file.endswith('.jpg'):
        continue
    n = random.uniform(0, 1)
    if (n > 0.9):
        shutil.copyfile('./TrainingData/'+file, targetFolder+file)
        fileName = os.path.splitext(file)[0]
        shutil.copyfile('./TrainingData/'+fileName+'.txt', targetFolder+fileName+'.txt')