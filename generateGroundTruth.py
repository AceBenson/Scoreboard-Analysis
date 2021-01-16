import cv2
import os
import json

def main():
    currentPath = os.getcwd()

    cap = cv2.VideoCapture("../AllVideos/videos/video1.mp4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )

    groundTruth_json = {
        "videoName": "video1", 
        "videoLength": length,
        "Score": [{"frameIdx": i, "groundTruth": []} for i in range(length)]
    }
    os.chdir(r"C:\Users\User\Desktop\Court Athena\Tesseract-OCR\images\video1_classified")
    for groundTruthFolder in os.listdir('./'):
        if not os.path.isdir(groundTruthFolder):
            continue
        print(groundTruthFolder)
        groundTruth = groundTruthFolder.split('_')
        for i in range(len(groundTruth)):
            groundTruth[i] = int(groundTruth[i])
        for frameFile in os.listdir('./'+groundTruthFolder):
            if not frameFile.endswith('.png'):
                continue
            # print(frameFile)
            frameIdx = os.path.splitext(frameFile)[0][5:]
            groundTruth_json["Score"][int(frameIdx)]["groundTruth"] = groundTruth
    with open("groundTruth.json", "w") as outfile:
        json.dump(groundTruth_json, outfile, indent=4)

if __name__ == '__main__':
    main()