# Face Detection & Face Segmentation

Dự án tập trung vào 2 bài toán thị giác máy tính liên quan đến khuôn mặt:

- Face Detection: phát hiện vị trí khuôn mặt trong ảnh đông người.
- Face Segmentation: tách chính xác vùng mặt (face mask) ở mức pixel.

Mục tiêu ứng dụng chính là camera an ninh và nhận diện khuôn mặt trong đám đông.

## Tổng quan nhanh

| Hạng mục | Nội dung |
|---|---|
| Chủ đề | Face Detection & Face Segmentation |
| Detection task | Khuôn mặt trong ảnh đông người |
| Segmentation task | Vùng mặt (face mask) |
| Dataset Detection | WIDER FACE |
| Dataset Segmentation | CelebAMask-HQ |
| Mô hình Detection | RetinaFace |
| Mô hình Segmentation | U-Net / Mask R-CNN |
| Use case | Camera an ninh, nhận diện khuôn mặt trong đám đông |

## Dataset sử dụng

### 1) WIDER FACE (cho Detection)
- Bộ dữ liệu chuẩn cho bài toán phát hiện khuôn mặt ngoài thực tế.
- Bao gồm nhiều bối cảnh khó: đông người, che khuất, góc nghiêng, ánh sáng phức tạp.

### 2) CelebAMask-HQ (cho Segmentation)
- Bộ dữ liệu phân vùng khuôn mặt chất lượng cao.
- Phù hợp để học mask chi tiết vùng mặt, tóc, da, và các thành phần khuôn mặt.

## Mô hình đề xuất

### Detection: RetinaFace
- Mạnh trên các trường hợp khuôn mặt nhỏ và đông người.
- Tối ưu cho bài toán giám sát và nhận diện trong môi trường thực tế.

### Segmentation: U-Net / Mask R-CNN
- U-Net: đơn giản, hiệu quả, dễ huấn luyện cho semantic segmentation.
- Mask R-CNN: mạnh hơn cho instance segmentation khi cần tách đối tượng rõ ràng.

## Cấu trúc thư mục

```
face-detection-segmentation/
├── README.md
├── requirements.txt
├── app/
├── data/
├── models/
├── notebooks/
└── src/
	├── detection/
	├── segmentation/
	└── utils/
```

## Cài đặt môi trường

Yêu cầu Python 3.9+.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Quy trình triển khai gợi ý

### Bước 1: Chuẩn bị dữ liệu
- Tải và tổ chức WIDER FACE vào thư mục data cho detection.
- Tải và tổ chức CelebAMask-HQ vào thư mục data cho segmentation.

### Bước 2: Huấn luyện detection
- Huấn luyện RetinaFace trên tập detection.
- Lưu trọng số vào thư mục models.

### Bước 3: Huấn luyện segmentation
- Huấn luyện U-Net hoặc Mask R-CNN trên CelebAMask-HQ.
- Lưu trọng số vào thư mục models.

### Bước 4: Suy luận và đánh giá
- Detection: đánh giá độ chính xác phát hiện khuôn mặt trong ảnh đông người.
- Segmentation: đánh giá chất lượng mask vùng mặt theo chỉ số IoU/Dice.

## Ứng dụng thực tế

- Hệ thống camera an ninh tại khu vực công cộng.
- Theo dõi và phân tích mật độ khuôn mặt trong đám đông.
- Tiền xử lý cho các hệ thống nhận diện khuôn mặt nâng cao.

## Ghi chú

- Thư mục data và models có thể rất lớn, nên không đưa trực tiếp lên GitHub.
- Sử dụng file .gitignore để tránh đẩy dữ liệu thô, trọng số mô hình và file tạm.
