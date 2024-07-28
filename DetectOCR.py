# from ultralytics import YOLO
# model = YOLO("best_ocr.pt")  # load a pretrained model (recommended for training)
# # results = model.predict(source="demo", conf=0.5,save=True, save_txt=True)
# results = model.predict(source="demo/bien-so-xe_0401085240.jpg", conf=0.5)
import math
from ultralytics import YOLO
# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

def detect_plate(results):
    listName = ['1', '2', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', '3', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', '0', '4', '5', '6', '7', '8', '9', 'A']

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
                if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
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
            for l in sorted(center_list, key=lambda x: x[0]):
                license_plate += str(l[2])

        return license_plate
    

model = YOLO('runs/detect/train2/weights/best_plate.pt')  
results = model(source=r'D:\OCR-2\bienxe2.png')
print(detect_plate(results))



