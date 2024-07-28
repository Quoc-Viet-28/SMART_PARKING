# SỬ DỤNG EPSILON ĐỂ TẠO THÀNH 1 ĐA GIÁC
import numpy as np
from ultralytics import YOLO
import numpy as np
import cv2
import math
from PIL import Image

model_2 = YOLO("runs/detect/train2/weights/best_plate.pt")


def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b


def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)


def detect_plate(results):
    listName = [
        "1",
        "2",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "K",
        "L",
        "M",
        "3",
        "N",
        "P",
        "S",
        "T",
        "U",
        "V",
        "X",
        "Y",
        "Z",
        "0",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
    ]
    for result in results:
        bb_list = result.boxes.numpy()
        names = result.names
        LP_type = "1"

        temp = ""
        center_list = []
        y_mean = 0
        y_sum = 0
        for bb in bb_list:
            xyxy = bb.xyxy
            x_c = (xyxy[0][0] + xyxy[0][2]) / 2
            y_c = (xyxy[0][1] + xyxy[0][3]) / 2
            cls = bb.cls
            y_sum += y_c
            center_list.append([x_c, y_c, listName[int(cls[0])]])

            # find 2 point to draw line
        if len(center_list) <= 0:
            return ""
        l_point = center_list[0]
        r_point = center_list[0]
        for cp in center_list:
            if cp[0] < l_point[0]:
                l_point = cp
            if cp[0] > r_point[0]:
                r_point = cp
        for ct in center_list:
            if l_point[0] != r_point[0]:
                if (
                    check_point_linear(
                        ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]
                    )
                    == False
                ):
                    LP_type = "2"

        y_mean = int(int(y_sum) / len(bb_list))

        # 1 line plates and 2 line plates
        line_1 = []
        line_2 = []
        license_plate = ""
        if LP_type == "2":
            for c in center_list:
                if int(c[1]) > y_mean:
                    line_2.append(c)
                else:
                    line_1.append(c)
            for l1 in sorted(line_1, key=lambda x: x[0]):
                license_plate += str(l1[2])
            license_plate += "-"
            for l2 in sorted(line_2, key=lambda x: x[0]):
                license_plate += str(l2[2])
        else:
            for i, l in enumerate(sorted(center_list, key=lambda x: x[0])):
                if i == 3:
                    license_plate += "-"
                    license_plate += str(l[2])
                else:
                    license_plate += str(l[2])
        return license_plate


image = cv2.imread(r"D:\OCR-2\2xe.jpg")
model = YOLO("plateNumberF-640.onnx", task="segment")
results = model(image)
segmented_images = []
Confidences = []
segment_confidence_pairs = []
if results is not None:
    for i, r in enumerate(results):
        if r.masks is not None:
            Masks = r.masks.xy
            for j, mask in enumerate(Masks):
                confidence = r.boxes.conf[j]
                Confidences.append(confidence)
                segment_confidence_pairs.append((mask, confidence))
        else:
            print("masks is None")
    segment_confidence_pairs.sort(key=lambda x: x[1], reverse=True)
    segmented_images = [segment for segment, _ in segment_confidence_pairs]
    for idx, segment in enumerate(segmented_images):
        segment_np = np.array(segment, dtype=np.float32)
        contours = [segment_np]
        if len(contours) > 0:
            for contour in contours:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                approx = approx.reshape(4, 2)
            print(approx)
            print(f"Confidence at index {idx}: {Confidences[idx]}")
            if len(approx) == 4:
                s = approx.sum(axis=1)
                topleft = approx[np.argmin(s)]
                botright = approx[np.argmax(s)]
                diff = np.diff(approx, axis=1)
                topright = approx[np.argmin(diff)]
                botleft = approx[np.argmax(diff)]
                approx = np.array(
                    [topleft, topright, botright, botleft], dtype=np.float32
                )
                widthA = np.sqrt(
                    ((botright[0] - botleft[0]) ** 2)
                    + ((botright[1] - botleft[1]) ** 2)
                )
                widthB = np.sqrt(
                    ((topright[0] - topleft[0]) ** 2)
                    + ((topright[1] - topleft[1]) ** 2)
                )
                heightA = np.sqrt(
                    ((topright[0] - botright[0]) ** 2)
                    + ((topright[1] - botright[1]) ** 2)
                )
                heightB = np.sqrt(
                    ((topleft[0] - botleft[0]) ** 2) + ((topleft[1] - botleft[1]) ** 2)
                )
                maxWidth = max(int(widthA), int(widthB))
                maxHeight = max(int(heightA), int(heightB))
                new_corners = np.array(
                    [
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1],
                    ],
                    dtype=np.float32,
                )
                matrix = cv2.getPerspectiveTransform(approx, new_corners)
                new_image = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
                output_file = f"segment_{idx}.jpg"
                cv2.imwrite(output_file, new_image)
                cv2.imshow(f"segment_{idx}", new_image)
                res_dec = model_2(source=f"segment_{idx}.jpg")
                print(detect_plate(res_dec))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
else:
    print("results is None")

for res in results:
    im_array = res.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()

