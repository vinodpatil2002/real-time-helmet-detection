from ultralytics import YOLO
import math
import cv2
import cvzone
import torch
from image_to_text import predict_number_plate
from paddleocr import PaddleOCR
import os

cap = cv2.VideoCapture("IMG_1058.MOV")  # For videos

model = YOLO("best.pt")

classNames = ["with helmet", "without helmet", "rider", "number plate"]
num = 0
old_npconf = 0

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory

# Folder to save cropped number plate images
save_folder = "number_plate_images"
os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist

# Set to store unique number plates
unique_number_plates = set()

while True:
    success, img = cap.read()
    # Check if the frame was read successfully
    if not success:
        break
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device="mps")
    for r in results:
        boxes = r.boxes
        li = dict()
        rider_box = list()
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy, confidences.unsqueeze(1), classes.unsqueeze(1)), 1)
        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
            # Get the indices of the rows where the value in column 1 is equal to 5.
            indices = torch.where(new_boxes[:, -1] == 2)
            # Select the rows where the mask is True.
            rows = new_boxes[indices]
            # Add rider details in the list
            for box in rows:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rider_box.append((x1, y1, x2, y2))
        except:
            pass
        for i, box in enumerate(new_boxes):
            # Bounding box
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box[4] * 100)) / 100
            # Class Name
            cls = int(box[5])
            if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or \
                    classNames[cls] == "number plate" and conf >= 0.5:
                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))
                if rider_box:
                    for j, rider in enumerate(rider_box):
                        if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
                                y2 <= rider_box[j][3]:
                            # highlight or outline objects detected by object detection models
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                            li.setdefault(f"rider{j}", [])
                            li[f"rider{j}"].append(classNames[cls])
                            if classNames[cls] == "number plate":
                                npx, npy, npw, nph, npconf = x1, y1, w, h, conf
                                crop = img[npy:npy + h, npx:npx + w]
                                try:
                                    # Check if the number plate is unique before saving
                                    number_plate_text, _ = predict_number_plate(crop, ocr)
                                    if number_plate_text and number_plate_text not in unique_number_plates:
                                        # Save the cropped number plate image to the specified folder
                                        cv2.imwrite(os.path.join(save_folder, f'number_plate_{num}.jpg'), crop)
                                        num += 1
                                        unique_number_plates.add(number_plate_text)
                                except Exception as e:
                                    print(e)
                        if li:
                            for key, value in li.items():
                                if key == f"rider{j}":
                                    if len(list(set(li[f"rider{j}"]))) == 3:
                                        try:
                                            vechicle_number, conf = predict_number_plate(crop, ocr)
                                            if vechicle_number and conf:
                                                cvzone.putTextRect(img, f"{vechicle_number} {round(conf*100, 2)}%",
                                                                   (x1, y1 - 50), scale=1.5, offset=10,
                                                                   thickness=2, colorT=(39, 40, 41),
                                                                   colorR=(105, 255, 255))
                                        except Exception as e:
                                            print(e)
        # Display the frame
        cv2.imshow('Video', img)
        li = list()
        rider_box = list()

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
