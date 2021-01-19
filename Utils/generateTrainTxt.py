import os

os.chdir(r"C:\Users\User\Desktop\Court Athena\Text-detection\darknet\build\darknet\x64\data")

with open("digits-train.txt", "w") as txtFile:
    for file in os.listdir("./digits"):
        if not file.endswith('.jpg'):
            continue
        print(file)
        txtFile.write("data/digits/"+file+"\n")