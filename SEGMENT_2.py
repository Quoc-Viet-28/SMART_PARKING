# #SỬ DỤNG EPSILON ĐỂ TẠO THÀNH 1 ĐA GIÁC
import numpy as np
from ultralytics import YOLO
from skimage import exposure
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


# image = cv2.imread(
#     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\train\images\a-1179-_jpg.rf.9805b274056af13c815aee1b1b998606.jpg"
# )
# # image = cv2.imread(
# #     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\train\images\a-746-_jpg.rf.510dae22cbfd9fbb6682c332aaf759b2.jpg"
# # )
# # image = cv2.imread(
# #     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\valid\images\a-75-_jpg.rf.12458f3fb8ce7270b5c570ebc86e7532.jpg"
# # )
# model = YOLO("best (5).pt", task="segment")
# results = model(image)
# segmented_images = []
# Confidences = []
# segment_confidence_pairs = []
# contours = []


# def get_segment_points(approx):
#     if len(approx) == 4:
#         approx = approx.reshape(4, 2)
#         print("bằng 4")
#         print(approx)
#         return approx
#     else:
#         print("khác 4")
#         x, y, w, h = cv2.boundingRect(approx)
#         approx = np.array(
#             [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
#         )
#         print(approx)
#         return approx


# if results is not None:
#     for i, r in enumerate(results):
#         if r.masks is not None:
#             Masks = r.masks.xy
#             for j, mask in enumerate(Masks):
#                 confidence = r.boxes.conf[j]
#                 Confidences.append(confidence)
#                 segment_confidence_pairs.append((mask, confidence))
#         else:
#             print("masks is None")
#     segment_confidence_pairs.sort(key=lambda x: x[1], reverse=True)
#     segmented_images = [segment for segment, _ in segment_confidence_pairs]
#     for idx, segment in enumerate(segmented_images):
#         segment_np = np.array(segment, dtype="float32")
#         contours = [segment_np]
#         # print(contours)
#     if len(contours) > 0:
#         for contour in contours:
#             peri = cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
#             approx = get_segment_points(approx)
#             approx_2 = np.zeros((4, 2), dtype="float32")
#             min_pair = approx[np.argmin(approx.sum(axis=1))]
#             print("Cặp có tổng nhỏ nhất:", min_pair)
#             max_difference_pair = approx[np.argmax(np.diff(approx, axis=1))]
#             print("Cặp có hiệu lớn nhất:", max_difference_pair)
#             topleft = approx[np.argmin(approx.sum(axis=1))]
#             botright = approx[np.argmax(approx.sum(axis=1))]
#             topright = approx[np.argmin(np.diff(approx, axis=1))]
#             botleft = approx[np.argmax(np.diff(approx, axis=1))]
#             # Extract individual points from the sorted array
#             # Find the other two points
#             other_points = [
#                 point
#                 for point in approx
#                 if not np.array_equal(point, min_pair)
#                 and not np.array_equal(point, max_difference_pair)
#             ]
#             botleft, botright = other_points
#             print("topleft:", topleft)
#             print("topright:", topright)
#             print("botleft:", botleft)
#             print("botright:", botright)
#             approx_2 = np.array(
#                 [topleft, topright, botright, botleft], dtype=np.float32
#             )
#             widthA = np.sqrt(
#                 ((botright[0] - botleft[0]) ** 2) + ((botright[1] - botleft[1]) ** 2)
#             )
#             widthB = np.sqrt(
#                 ((topright[0] - topleft[0]) ** 2) + ((topright[1] - topleft[1]) ** 2)
#             )
#             heightA = np.sqrt(
#                 ((topright[0] - botright[0]) ** 2) + ((topright[1] - botright[1]) ** 2)
#             )
#             heightB = np.sqrt(
#                 ((topleft[0] - botleft[0]) ** 2) + ((topleft[1] - botleft[1]) ** 2)
#             )
#             print("WidthA:", widthA)
#             print("WidthB:", widthB)
#             print("heightA:", heightA)
#             print("heightB:", heightB)
#             maxWidth = max(int(widthA), int(widthB))
#             maxHeight = max(int(heightA), int(heightB))
#             new_corners = np.array(
#                 [
#                     [0, 0],
#                     [maxWidth - 1, 0],
#                     [maxWidth - 1, maxHeight - 1],
#                     [0, maxHeight - 1],
#                 ],
#                 dtype=np.float32,
#             )
#             print("Max Width:", maxWidth)
#             print("Max Height:", maxHeight)
#             print("Corner Points:", new_corners)
#             matrix = cv2.getPerspectiveTransform(approx_2, new_corners)
#             new_image = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

#             output_file = f"segment_{idx}.jpg"
#             cv2.imwrite(output_file, new_image)
#             cv2.imshow(f"segment_{idx}", new_image)
#             # cv2.imshow(f"segment", new_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

#     else:
#         print("results is None")


# image = cv2.imread(
#     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\train\images\a-1179-_jpg.rf.9805b274056af13c815aee1b1b998606.jpg"
# )
# image = cv2.imread(
#     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\train\images\a-746-_jpg.rf.510dae22cbfd9fbb6682c332aaf759b2.jpg"
# )
# )
# image = cv2.imread(
#     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\valid\images\a-75-_jpg.rf.12458f3fb8ce7270b5c570ebc86e7532.jpg"
# )
# image = cv2.imread(
#     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\train\images\a-1676-_jpg.rf.0f5fcdeacd8f5b0cd3150225283934ef.jpg"
# )
# image = cv2.imread(
#     r"D:\dev-viet\yolov8\OCR-2\PlateNumber-4\train\images\a-1628-_jpg.rf.3acdd746a9c1755ed8b2f359fa7ac802.jpg"
# )
image = cv2.imread(
    r"D:\OCR-2\TEST\bienxemay.png"
)

model = YOLO("D:\OCR-2\plateNumberF-640.onnx", task="segment")
results = model(image)
segmented_images = []
Confidences = []
segment_confidence_pairs = []
contours = []


def get_segment_points(approx):
    if len(approx) == 4:
        approx = approx.reshape(4, 2)
        print("bằng 4")
        print(approx)
        return approx
    else:
        print("khác 4")
        x, y, w, h = cv2.boundingRect(approx)
        approx = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )
        print(approx)
        return approx


def do_lines_intersect(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if (
        (line1[0] == line2[0]).all()
        or (line1[0] == line2[1]).all()
        or (line1[1] == line2[0]).all()
        or (line1[1] == line2[1]).all()
    ):
        return False
    else:
        if det == 0:
            return False
        else:
            intersection_x = (
                (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
            ) / det
            intersection_y = (
                (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
            ) / det

            if (
                min(x1, x2) <= intersection_x <= max(x1, x2)
                and min(y1, y2) <= intersection_y <= max(y1, y2)
                and min(x3, x4) <= intersection_x <= max(x3, x4)
                and min(y3, y4) <= intersection_y <= max(y3, y4)
            ):
                return True
            else:
                return False


def find_intersecting_lines(coordinates):
    intersecting_lines = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            line1 = [coordinates[i], coordinates[j]]
            for k in range(len(coordinates)):
                for l in range(k + 1, len(coordinates)):
                    line2 = [coordinates[k], coordinates[l]]
                    if do_lines_intersect(line1, line2):
                        intersecting_lines.append((line1, line2))
    return intersecting_lines[0]


def find_top_and_bottom_coordinates(diagonalline):
    tl = []
    tr = []
    bl = []
    br = []
    coord1, coord2 = diagonalline[0]
    coord3, coord4 = diagonalline[1]
    data = [coord1, coord2, coord3, coord4]
    temp = 999999
    idex = 0
    i = 0
    for item in data:
        if item[1] < temp:
            idex = i
            temp = item[1]
        i += 1

    if idex == 1:
        coord1 = data[1]
        coord2 = data[0]

    if idex == 3:
        coord3 = data[3]
        coord4 = data[2]

    if idex == 0 or idex == 1:
        print("0-----1")
        x1 = coord1[0] - coord2[0]
        # nghien ben phai
        if x1 < 0:
            print("x-----1")
            tl = coord1
            br = coord2
            if coord3[0] < coord4[0]:
                print("y-----1")
                bl = coord3
                tr = coord4
            else:
                print("z-----1")
                bl = coord4
                tr = coord3
        # nghien ben trai
        else:
            print("m-----1")
            tr = coord1
            bl = coord2
            if coord3[0] < coord4[0]:
                # pass
                print("zz----12")
                br = coord4
                tl = coord3
            else:
                print("n-----1")
                br = coord3
                tl = coord4
    else:
        print("2------3")
        x1 = coord3[0] - coord4[0]
        # nghien ben phai
        if x1 < 0:
            print("4-----5")
            tl = coord3
            br = coord4
            if coord1[0] < coord2[0]:
                # pass
                print("11----12")
                bl = coord1
                tr = coord2
            else:
                print("p-----1")
                bl = coord2
                tr = coord1
        else:
            print("6---7")
            tr = coord3
            bl = coord4
            if coord1[0] < coord2[0]:
                print("89---7")
                # pass
                br = coord2
                tl = coord1
            else:
                print("10----11")
                br = coord1
                tl = coord2
    return tl, tr, bl, br


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
        segment_np = np.array(segment, dtype="float32")
        contours = [segment_np]
        if len(contours) > 0:
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                approx = get_segment_points(approx)
                approx_2 = np.abs(np.zeros((4, 2), dtype="float32"))
                diagonalline = find_intersecting_lines(approx)
                topleft, topright, botleft, botright = find_top_and_bottom_coordinates(
                    diagonalline
                )
                approx_2 = np.array(
                    [topleft, topright, botright, botleft], dtype=np.float32
                )
                widthA = np.sqrt(
                    ((botright[0] - botleft[0]) ** 2) + ((botright[1] - botleft[1]) ** 2)
                )
                widthB = np.sqrt(
                    ((topright[0] - topleft[0]) ** 2) + ((topright[1] - topleft[1]) ** 2)
                )
                heightA = np.sqrt(
                    ((topright[0] - botright[0]) ** 2) + ((topright[1] - botright[1]) ** 2)
                )
                heightB = np.sqrt(
                    ((topleft[0] - botleft[0]) ** 2) + ((topleft[1] - botleft[1]) ** 2)
                )
                print("WidthA:", widthA)
                print("WidthB:", widthB)
                print("heightA:", heightA)
                print("heightB:", heightB)
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
                print("Max Width:", maxWidth)
                print("Max Height:", maxHeight)
                print("Corner Points:", new_corners)
                matrix = cv2.getPerspectiveTransform(approx_2, new_corners)
                new_image = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

                output_file = f"segment_{idx}.jpg"
                cv2.imwrite(output_file, new_image)
                cv2.imshow(f"segment_{idx}", new_image)
                res = model_2(new_image)
                print(detect_plate(res))
                # cv2.imshow(f"segment", new_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

else:
    print("results is None")
for res in results:
    im_array = res.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
