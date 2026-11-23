import cv2
import random
import argparse
import numpy as np


def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

    key = 1
    frame = None
    cap = None

    if source_file == "0":
        image_type = False
        source_file = 0
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats

    if image_type:
        frame = cv2.imread(source_file)
        key = 0
    else:
        cap = cv2.VideoCapture(source_file)

    return image_type, key, frame, cap


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="data/images/horses.jpg",
        help="Image/Video Source"
    )
    parser.add_argument(
        "--names",
        type=str,
        default="data/class.names",
        help="Class Names"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.onnx",
        help="ONNX Model"
    )
    parser.add_argument(
        "--tresh",
        type=float,
        default=0.25,
        help="Confidence Threshold"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Bounding Box Thickness"
    )

    args = parser.parse_args()

    # print("Loading model:", args.model)

    model = cv2.dnn.readNet(args.model)

    IMAGE_SIZE = 640

    with open(args.names, "r") as f:
        NAMES = [line.strip() for line in f.readlines()]

    COLORS = [
        [random.randint(0, 255) for _ in range(3)]
        for _ in NAMES
    ]

    image_type, key, frame, cap = loadSource(args.source)

    grabbed = True

    while True:

        if not image_type:
            grabbed, frame = cap.read()

        if not grabbed or frame is None:
            break

        image = frame.copy()

        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (IMAGE_SIZE, IMAGE_SIZE),
            swapRB=True,
            crop=False
        )

        model.setInput(blob)

        preds = model.forward()

        # print("Raw output shape:", preds.shape)

        preds = preds.transpose((0, 2, 1))

        # print("Transposed shape:", preds.shape)

        image_height, image_width = image.shape[:2]

        x_factor = image_width / IMAGE_SIZE
        y_factor = image_height / IMAGE_SIZE

        class_ids = []
        confs = []
        boxes = []

        rows = preds[0].shape[0]

        for i in range(rows):

            row = preds[0][i]

            # YOLOv8 classes start at index 4
            classes_scores = row[4:]

            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            if confidence > args.tresh:

                x = row[0]
                y = row[1]
                w = row[2]
                h = row[3]

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confs.append(float(confidence))
                class_ids.append(int(class_id))

        # print(f"Boxes before NMS: {len(boxes)}")

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confs,
            args.tresh,
            0.45
        )

        if len(indexes) > 0:

            indexes = np.array(indexes).flatten()

            # print(f"Boxes after NMS: {len(indexes)}")

            for i in indexes:

                left, top, width, height = boxes[i]

                class_id = class_ids[i]
                confidence = confs[i]

                color = COLORS[class_id]

                cv2.rectangle(
                    image,
                    (left, top),
                    (left + width, top + height),
                    color,
                    args.thickness
                )

                label = f"{NAMES[class_id]} {confidence:.2f}"

                cv2.putText(
                    image,
                    label,
                    (left, max(top - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.imshow("Detected", image)

        if cv2.waitKey(key) & 0xFF == ord('q'):
            break

        if image_type:
            cv2.waitKey(0)
            break

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()