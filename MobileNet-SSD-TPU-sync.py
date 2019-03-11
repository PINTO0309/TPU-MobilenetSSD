import argparse
import platform
import numpy as np
import cv2
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine


# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", help="Path of the detection model.")
    parser.add_argument("--label", default="coco_labels.txt", help="Path of the labels file.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")
    args = parser.parse_args()

    fps = ""
    detectfps = ""
    framecount = 0
    detectframecount = 0
    time1 = 0
    time2 = 0
    box_color = (255, 128, 0)
    box_thickness = 1
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)
    percentage = 0.0

    camera_width = 320
    camera_height = 240

    cap = cv2.VideoCapture(args.usbcamno)
    cap.set(cv2.CAP_PROP_FPS, 150)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    # Initialize engine.
    engine = DetectionEngine(args.model)
    labels = ReadLabelFile(args.label) if args.label else None

    while True:
        t1 = time.perf_counter()

        ret, color_image = cap.read()
        if not ret:
            break

        # Run inference.
        prepimg = color_image[:, :, ::-1].copy()
        prepimg = Image.fromarray(prepimg)

        tinf = time.perf_counter()
        ans = engine.DetectWithImage(prepimg, threshold=0.5, keep_aspect_ratio=True, relative_coord=False, top_k=10)
        print(time.perf_counter() - tinf, "sec")


        # Display result.
        if ans:
            detectframecount += 1
            for obj in ans:
                box = obj.bounding_box.flatten().tolist()
                box_left = int(box[0])
                box_top = int(box[1])
                box_right = int(box[2])
                box_bottom = int(box[3])
                cv2.rectangle(color_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

                percentage = int(obj.score * 100)
                label_text = labels[obj.label_id] + " (" + str(percentage) + "%)" 

                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(color_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(color_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

        cv2.putText(color_image, fps,       (camera_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(color_image, detectfps, (camera_width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

        cv2.namedWindow('USB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('USB Camera', color_image)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime

if __name__ == '__main__':
    main()
