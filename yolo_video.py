import cv2
import argparse
import numpy as np

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label + f" {confidence:.2f}", (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', default='yolov3.cfg', help='path to yolo config file')
    ap.add_argument('-w', '--weights', default='yolov3.weights', help='path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', default='yolov3.txt', help='path to text file containing class names')
    args = ap.parse_args()

    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNet(args.weights, args.config)

    # Initialize video stream
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("[INFO] starting video stream...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392

        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        
        # Forward pass
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)

        cv2.imshow("Object Detection (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
