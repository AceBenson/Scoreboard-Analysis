import argparse
import os
import glob
import time
import cv2
import json
import numpy as np

from DigitsDetection import darknet
from DigitsRecognition.scoreboardOCR import scoreboardOCR

# scoreboardPos = [0, 620, 680, 100]
scoreboardPos = [75, 580, 300, 80]

def parser():
    parser = argparse.ArgumentParser(description="Scoreboard Analysis")
    parser.add_argument("--input", type=str, default=r"../AllVideos/testVideos/video4.mp4",
                        help="input video")

    # Scoreboard Detection                        
    
    # Digits Detection
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="./DigitsDetection/weights/yolo-obj_final.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./DigitsDetection/cfg/yolo-obj.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./DigitsDetection/data/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")

    # Digits Recognition


    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))

def digitsDetection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    # Filter top 4 confidence
    detections.sort(key=lambda x: x[1], reverse=True)
    detections = detections[:4]

    h, w, _ = image.shape
    # Deal with detection result
    cropData = []
    for label, confidence, bbox in detections:
        xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)
        xmin = max(0, int(xmin*image.shape[1]/image_resized.shape[1])-1)
        ymin = max(0, int(ymin*image.shape[0]/image_resized.shape[0])-1)
        xmax = min(w, int(xmax*image.shape[1]/image_resized.shape[1])+1)
        ymax = min(h, int(ymax*image.shape[0]/image_resized.shape[0])+1)
        cropData.append([xmin, ymin, xmax, ymax])

    # Make sure data is set1, set2, score1, score2
    cropData.sort(key=lambda data: data[0]+data[1])
    return cropData

def main():
    args = parser()
    check_arguments_errors(args)

    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    cap = cv2.VideoCapture(args.input)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    predictResult_json = {
        "videoName": args.input, 
        "videoLength": length,
        "Score": [{"frameIdx": i, "predictResult": []} for i in range(length)]
    }

    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        # scoreboardPos = [75, 580, 300, 80]
        # scoreboardPos = [0, 640, 520, 80]
        x, y, w, h = scoreboardPos
        frame = frame[y:y+h, x:x+w]

        # Digits Detection
        cropData = digitsDetection(frame, network, class_names, class_colors, args.thresh)

        # Digits Recognition
        predictedTexts = scoreboardOCR(frame, cropData)
        predictResult_json["Score"][index]["predictResult"] = predictedTexts
        print(predictedTexts)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        index = index+1
    with open("PredictResult/predictResult.json", "w") as outfile:
        json.dump(predictResult_json, outfile, indent=4)

if __name__ == "__main__":
    main()