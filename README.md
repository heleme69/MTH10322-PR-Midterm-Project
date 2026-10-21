# Nhận Dạng Chữ Số Viết Tay MNIST - Four Classifiers from Scratch

Đồ án Giữa kỳ môn Nhận dạng Mẫu và Học máy (MTH10322). Cài đặt và đánh giá bốn mô hình phân lớp từ đầu (*from-scratch*) trên tập dữ liệu chuẩn MNIST (28 × 28, d = 784, K = 10), tích hợp ứng dụng giao diện web Gradio phục vụ thử nghiệm trực quan.

## 1. Phân công công việc

| STT | Thành viên | MSSV | Vai trò | Trách nhiệm kỹ thuật | Đóng góp |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | (to be updated) | (to be updated) | Nhóm trưởng | Điều phối repo; Cài đặt mạng neuron MLP (Backpropagation, SGD Momentum); Hoàn thiện pipeline thực nghiệm và báo cáo LaTeX. | 100% |
| 2 | (to be updated) | (to be updated) | Thành viên | Xây dựng module dữ liệu (`dataset.py`); Cài đặt Multinomial Naïve Bayes và Softmax Regression; Khảo sát tham số hội tụ loss. | 100% |
| 3 | (to be updated) | (to be updated) | Thành viên | Cài đặt Random Forest (Bagging, Subsampling); Xây dựng module đánh giá (`metrics.py`), xuất ma trận nhầm lẫn; Phát triển web demo Gradio. | 100% |

## 2. Kết quả thực nghiệm

Đánh giá trên 10.000 mẫu kiểm thử chuẩn MNIST:

| Mô hình | Cơ sở thuật toán | Accuracy (%) | Macro F1 | Thời gian Train (s) | Thời gian Test (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Multinomial Naïve Bayes | Xác suất độc lập có điều kiện | 83.60% | 0.8345 | 0.1s | 0.05s |
| Softmax Regression | Phân lớp tuyến tính đa lớp | 92.03% | 0.9192 | 6.7s | 0.02s |
| Random Forest | Tập hợp cây quyết định (Bagging) | 95.89% | 0.9585 | 31.7s | 0.85s |
| MLP from Scratch | Mạng neuron sâu (2 Hidden Layers) | 97.92% | 0.9791 | 18.1s | 0.12s |

## 3. Cấu trúc mã nguồn

```text
MTH10322-PR-Midterm-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── figures/               # Biểu đồ phân tích, ma trận nhầm lẫn
│   ├── main.tex               # Báo cáo kỹ thuật LaTeX (XeLaTeX)
│   └── sections/              # Các phần nội dung báo cáo
├── notebooks/
│   └── mnist_experiments.ipynb # Pipeline thực nghiệm tổng thể
└── src/
    ├── dataset.py             # Tiền xử lý và phân chia tập dữ liệu
    ├── metrics.py             # Đánh giá ma trận nhầm lẫn, F1-score
    ├── demo_app.py            # Giao diện web Gradio tương tác
    └── models/
        ├── naive_bayes.py
        ├── softmax_regression.py
        ├── mlp.py
        └── ensemble.py
```

## 4. Hướng dẫn thực thi

### Chạy cục bộ (Local)

1. Sao chép kho lưu trữ:
   ```bash
   git clone https://github.com
   cd MTH10322-PR-Midterm-Project
   ```

2. Tạo và kích hoạt môi trường ảo:
   * Windows (PowerShell):
     ```powershell
     python -m venv venv
     venv\Scripts\Activate.ps1
     ```
   * macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Cài đặt các gói phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

4. Khởi chạy thực nghiệm hoặc ứng dụng tương tác:
   * Mở Jupyter Lab:
     ```bash
     jupyter lab notebooks/mnist_experiments.ipynb
     ```
   * Chạy ứng dụng Gradio vẽ tay:
     ```bash
     python src/demo_app.py
     ```
