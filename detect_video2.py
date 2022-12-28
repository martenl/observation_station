""""
TEST_DATA=../all_models
Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
  
  python3 examples/small_object_detection.py \
>   --model test_data/ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite \
>   --label test_data/coco_labels.txt \
>   --input test_data/kite_and_cold.jpg \
>   --tile_size 1352x900,500x500,250x250 \
>   --tile_overlap 50 \
>   --score_threshold 0.5 \
>   --output ${HOME}/object_detection_results.jpg
"""
import argparse
import time

import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


def main():
    # default_model_dir = '../all_models'
    # default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    # default_labels = 'coco_labels.txt'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', help='.tflite model path',
    #                     default=os.path.join(default_model_dir,default_model))
    # parser.add_argument('--labels', help='label file path',
    #                     default=os.path.join(default_model_dir, default_labels))
    # parser.add_argument('--top_k', type=int, default=3,
    #                     help='number of categories with highest score to display')
    # parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    # parser.add_argument('--threshold', type=float, default=0.1,
    #                     help='classifier score threshold')
    # args = parser.parse_args()

    # print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter('../test_data/ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite')  # args.model)
    interpreter.allocate_tensors()
    labels = read_label_file('../test_data/coco_labels.txt')  # args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture('../test_data/bike_ride.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    frame_count = 0
    time_stamp = time.time_ns()
    found_labels = set()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, 0.5)[:3]
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
        out.write(cv2_im)
        for obj in objs:
            found_labels.add(labels.get(obj.id, obj.id))

        for i in range(fps):
            ret, frame = cap.read()
            if not ret:
                break
            frame = append_objs_to_img(frame, inference_size, objs, labels)
            out.write(frame)

        frame_count = frame_count + 1
        if frame_count % 1000 == 0:
            percentage = (frame_count / length) * 100
            print(percentage, "% done")
            elapsed_timein_sec = (time.time_ns() - time_stamp) / (1000 * 1000)
            print(elapsed_timein_sec, "seconds")
            print("Found labels: ", found_labels)
    out.release()
    cap.release()


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


if __name__ == '__main__':
    main()
