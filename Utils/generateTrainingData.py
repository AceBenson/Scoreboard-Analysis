import cv2
import os
import random
import json

video1CropData = [[229, 27, 14, 18], [219, 49, 14, 18], [254, 25, 24, 19], [245, 48, 24, 19]]

def main():
    cap = cv2.VideoCapture("../../AllVideos/videos/video1.mp4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )

    with open(r'C:\Users\User\Desktop\Court Athena\Tesseract-OCR\images\video1_classified\groundTruth.json') as jsonFile:
        groundTruthJson = json.load(jsonFile)
        index = 0
        while cap.isOpened:
            ret, frame = cap.read()
            if ret == False:
                break
            # if index < 1000:
            #     index = index+1
            #     continue
            scoreboardPos = [75, 580, 300, 80]
            x, y, w, h = scoreboardPos
            scoreboardCenterX = int(x+w/2)
            scoreboardCenterY = int(y+h/2)
            newW = random.randint(300, 450)
            newH = random.randint(80, 200)
            # newW = w
            # newH = h
            newX = int(scoreboardCenterX - newW/2)
            newY = int(scoreboardCenterY - newH/2)
            diffX = x - newX
            diffY = y - newY

            scoreboard = frame[newY:newY+newH, newX:newX+newW]
            # scoreboard = frame[y:y+h, x:x+w]

            cropData = video1CropData
            # print(len(cropData))

            if len(groundTruthJson["Score"][index]["groundTruth"]) != 0:
                cv2.imwrite("TrainingData/frame" + str(index) + ".jpg", scoreboard)
                txtContent = []
                for idx, data in enumerate(cropData):
                    Dx, Dy, Dw, Dh = data
                    Dx = Dx + diffX
                    Dy = Dy + diffY
                    NumberClass = groundTruthJson["Score"][index]["groundTruth"][idx]
                    # if NumberClass >= 10:
                    #     leftDw = int(Dw/2)-2
                    #     scoreboard = cv2.rectangle(scoreboard, (Dx, Dy), (Dx+leftDw, Dy+Dh), (0, 255, 0), 1)
                    #     row = [0 for i in range(5)]
                    #     row[0] = NumberClass/10
                    #     row[1] = (Dx + leftDw/2) / newW
                    #     row[2] = (Dy + Dh/2) / newH
                    #     row[3] = leftDw / newW
                    #     row[4] = Dh / newH
                    #     txtContent.append(' '.join(map(str, row)) + "\n")

                    #     Dx = Dx + leftDw
                    #     rightDw = Dw - leftDw
                    #     scoreboard = cv2.rectangle(scoreboard, (Dx, Dy), (Dx+rightDw, Dy+Dh), (0, 255, 0), 1)
                    #     row = [0 for i in range(5)]
                    #     row[0] = NumberClass%10
                    #     row[1] = (Dx + rightDw/2) / newW
                    #     row[2] = (Dy + Dh/2) / newH
                    #     row[3] = rightDw / newW
                    #     row[4] = Dh / newH
                    #     txtContent.append(' '.join(map(str, row)) + "\n")
                    # else:
                    scoreboard = cv2.rectangle(scoreboard, (Dx, Dy), (Dx+Dw, Dy+Dh), (0, 255, 0), 1)
                    row = [0 for i in range(5)]
                    row[0] = 0
                    row[1] = (Dx + Dw/2) / newW
                    row[2] = (Dy + Dh/2) / newH
                    row[3] = Dw / newW
                    row[4] = Dh / newH
                    txtContent.append(' '.join(map(str, row)) + "\n")
                with open("TrainingData/frame" + str(index) + ".txt", "w") as txtFile:
                    txtFile.writelines(txtContent)

            cv2.imshow('scoreboard', scoreboard)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            index = index+1

if __name__ == "__main__":
    main()