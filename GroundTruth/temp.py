import json

fileName = './video4/groundTruth.json'

groundTruthJsonFile = open(fileName, 'r')
groundTruthJson = json.load(groundTruthJsonFile)

for index in range(groundTruthJson["videoLength"]):
    groundTruth = groundTruthJson["Score"][index]["groundTruth"]
    groundTruth = list(map(str, groundTruth))
    groundTruthJson["Score"][index]["groundTruth"] = groundTruth
groundTruthJsonFile.close()

groundTruthJsonFile = open(fileName, 'w')
json.dump(groundTruthJson, groundTruthJsonFile, indent=4)
groundTruthJsonFile.close()