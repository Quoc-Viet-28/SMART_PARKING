from ultralytics import YOLO

# Định nghĩa danh sách các ký tự có thể xuất hiện trong biển số xe
listName = ['1', '2', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', '3', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', '0', '4', '5', '6', '7', '8', '9', 'A']
#LISTNAME = ['0','1','2','3','4','5','6','7','8','9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
# Tạo mô hình YOLO và thực hiện phát hiện trên hình ảnh
model = YOLO('runs/detect/train2/weights/best.pt')
results = model(source='biensoxe2.png')

# Khởi tạo biến để lưu trữ biển số xe
license_plate = ""

# Lặp qua các kết quả để xác định lớp và tọa độ của từng ký tự
char_info = []  # Danh sách thông tin về các ký tự

for result in results:
    bb_list = result.boxes.numpy()
    names = result.names

    for bb in bb_list:
        xyxy = bb.xyxy
        cls = bb.cls  # Lớp của ký tự
        print("lớp thứ: ", int(cls[0]))
        class_name = names[int(cls[0])]  # Xác định tên lớp từ danh sách listName
        print("giá trị lớp: ", class_name)
        # Thêm thông tin về ký tự vào danh sách char_info
        char_info.append({"class_name": class_name, "x_center": (xyxy[0][0] + xyxy[0][2]) / 2})
        print("tọa độ: ", (xyxy[0][0] + xyxy[0][2]) / 2)

# Sắp xếp danh sách char_info dựa trên tọa độ x_center
sorted_chars = sorted(char_info, key=lambda x: x["x_center"], reverse=False)

# Tạo chuỗi biển số xe từ các ký tự đã sắp xếp
for char in sorted_chars:
    license_plate += char["class_name"]

# In ra biển số xe đã sắp xếp từ trái qua phải
print("Biển số xe:", license_plate)


