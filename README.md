# YOLOv8 Object Detection with OpenCV

This project runs YOLOv8 object detection on images, video files, or a webcam using OpenCV's DNN module and an ONNX model. Sample images, a road video, and the 80 COCO class names are included.

Course: [YOLO Masterclass: Deep Learning & Computer Vision](https://www.udemy.com/course/yolo-masterclass-deep-learning-computer-vision-course/).

## Setup

Use Python 3.12 or newer, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Official model artifacts are not included. Provide a compatible YOLOv8 ONNX export with 640-pixel input dimensions. One way to create it is to install Ultralytics and run:

```bash
python -m pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
```

Copy the resulting ONNX file into this directory, or pass its path with `--model`.

## Run

```bash
python detection.py --source data/images/zidane.jpg --model yolov8n.onnx
python detection.py --source data/videos/road.mp4 --model yolov8n.onnx
python detection.py --source 0 --model yolov8n.onnx
```

The default image is `data/images/horses.jpg`. Press `q` to stop video or webcam inference; image inference pauses until a key is pressed.

Options include `--source`, `--names`, `--model`, `--tresh` (confidence threshold), and `--thickness`. A desktop display is required for OpenCV's result window.
