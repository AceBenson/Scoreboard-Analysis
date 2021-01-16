import os

os.chdir(r"C:\Users\User\Desktop\Court Athena\Text-detection\darknet\build\darknet\x64\data")

with open("0-9-train.txt", "w") as txtFile:
    for file in os.listdir("./0-9"):
        if not file.endswith('.jpg'):
            continue
        print(file)
        txtFile.write("data/0-9/"+file+"\n")