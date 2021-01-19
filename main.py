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

    parser.add_argument("-savePredictResult", required=False, action="store_true")
    parser.add_argument("-drawBBox", required=False, action="store_true")

    # Scoreboard Detection                        
    parser.add_argument("--ScoreboardDetection_weights", default="./ScoreboardDetection/weights/yolov3-tiny-prn-custom_last.weights")
    parser.add_argument("--ScoreboardDetection_cfg", default="./ScoreboardDetection/cfg/yolov3-tiny-prn-custom.cfg")
    parser.add_argument("--ScoreboardDetection_data", default="./ScoreboardDetection/data/tabletennis.data")

    # Digits Detection
    parser.add_argument("--DigitsDetection_weights", default="./DigitsDetection/weights/yolo-obj_final.weights")
    parser.add_argument("--DigitsDetection_cfg", default="./DigitsDetection/cfg/yolo-obj.cfg")
    parser.add_argument("--DigitsDetection_data", default="./DigitsDetection/data/obj.data")

    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")

    # Digits Recognition


    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.ScoreboardDetection_cfg):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.ScoreboardDetection_weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.ScoreboardDetection_data):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if not os.path.exists(args.DigitsDetection_cfg):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.DigitsDetection_weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.DigitsDetection_data):
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

def scoreboardDetection(image, network, class_names, class_colors, thresh):
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

    h, w, _ = image.shape

    for label, confidence, bbox in detections:
        if label == "ScoreBoard":
            xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)
            xmin = max(0, int(xmin*image.shape[1]/image_resized.shape[1])-1)
            ymin = max(0, int(ymin*image.shape[0]/image_resized.shape[0])-1)
            xmax = min(w, int(xmax*image.shape[1]/image_resized.shape[1])+1)
            ymax = min(h, int(ymax*image.shape[0]/image_resized.shape[0])+1)
            return [xmin, xmax, ymin, ymax]
    return None

def main():
    args = parser()
    check_arguments_errors(args)

    digitsDetection_network, digitsDetection_class_names, digitsDetection_class_colors = darknet.load_network(
        args.DigitsDetection_cfg,
        args.DigitsDetection_data,
        args.DigitsDetection_weights,
        batch_size=args.batch_size
    )

    scoreboardDetection_network, scoreboardDetection_class_names, scoreboardDetection_class_colors = darknet.load_network(
        args.ScoreboardDetection_cfg,
        args.ScoreboardDetection_data,
        args.ScoreboardDetection_weights,
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

        # Scorebooard Detection

        # scoreboardPos = [75, 580, 300+75, 80+580]
        scoreboardPos = scoreboardDetection(frame, scoreboardDetection_network, scoreboardDetection_class_names, scoreboardDetection_class_colors, args.thresh)
        if scoreboardPos != None:
            # print(scoreboardPos)
            xmin, xmax, ymin, ymax = scoreboardPos
            frame = frame[ymin:ymax, xmin:xmax]

            # Digits Detection
            cropData = digitsDetection(frame, digitsDetection_network, digitsDetection_class_names, digitsDetection_class_colors, args.thresh)

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