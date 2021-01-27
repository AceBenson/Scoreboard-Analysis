import json

def main():
    groundTruthJsonFile = open('../GroundTruth/video3/groundTruth.json')
    groundTruthJson = json.load(groundTruthJsonFile)
    predictResultJsonFile = open('../PredictResult/video3/predictResult.json')
    predictResultJson = json.load(predictResultJsonFile)

    if predictResultJson['videoLength'] != groundTruthJson['videoLength']:
        print("Different video length, predictResult:", predictResultJson['videoLength'], "groundTruth: ", groundTruthJson['videoLength'])
    
    print("videoName: ", groundTruthJson['videoName'])
    Total = 0
    Correct = 0
    for index in range(groundTruthJson['videoLength']):
        if index % 10 != 0: 
            continue

        groundTruth = groundTruthJson["Score"][index]["groundTruth"]
        predictResult = predictResultJson["Score"][index]["predictResult"]

        if len(groundTruth) == 0:
            continue

        if groundTruth == predictResult:
            Correct = Correct + 1
        else:
            print("Index:", index, "  Ground Truth: ", groundTruth, "  Predict Result: ", predictResult)
        Total = Total+1

    print("Correct = ", Correct)
    print("Total = ", Total)
    print("Correct/Total = ", Correct/Total)

    groundTruthJsonFile.close()
    predictResultJsonFile.close()

if __name__ == '__main__':
    main()
    