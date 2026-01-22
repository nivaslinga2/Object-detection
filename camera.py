import cv2
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        self.classes = []
        with open('yolov3.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.output_layers = self.get_output_layers(self.net)

    def __del__(self):
        self.video.release()

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.colors[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label + f" {confidence:.2f}", (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

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

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                self.draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
