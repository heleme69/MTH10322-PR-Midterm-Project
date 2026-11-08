# Nhận dạng chữ số viết tay MNIST bằng bốn phương pháp phân lớp

**Đồ án 1 (giữa kỳ) – Môn Nhận dạng**  
Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Giảng viên:** TS. Trương Phước Hưng

| Thành viên | MSSV |
|---|---|
| Lê Huy | 24110022 |
| Trương Hoàng Phúc | 24280020 |
| Trần Phương Thảo | 24110242 |

## 1. Tổng quan

Dự án triển khai và so sánh bốn phương pháp phân lớp chữ số viết tay trên bộ dữ liệu **MNIST**:

| Phương pháp | Triển khai |
|---|---|
| Multinomial Naive Bayes | NumPy (`src/models/naive_bayes.py`) |
| Softmax Regression | NumPy (`src/models/softmax_regression.py`) |
| Multilayer Perceptron (MLP) | PyTorch (`src/models/mlp.py`) |
| Random Forest | NumPy và scikit-learn (`src/models/ensemble.py`) |

Naive Bayes và Softmax Regression được tự triển khai bằng NumPy. MLP tự thực hiện lan truyền ngược và cập nhật tham số, sử dụng PyTorch cho phép toán tensor và lớp tuyến tính. Random Forest tự triển khai bootstrap và majority voting, sử dụng `DecisionTreeClassifier` làm cây cơ sở.

Dữ liệu được chia thành **50.000 ảnh train, 10.000 ảnh validation và 10.000 ảnh test**. Siêu tham số được chọn trên validation; tập test dùng để đánh giá cuối cùng.

> **Dành cho người chấm:** notebook đã được chạy sẵn và lưu kèm kết quả, mọi file đầu ra và model đã huấn luyện đều có trong repository. Có thể xem kết quả ngay (mục 3) hoặc chạy demo vẽ chữ số (mục 4) mà không cần huấn luyện lại.

## 2. Cấu trúc dự án

```text
MTH10322-PR-Midterm-Project/
├── artifacts/                  # model đã huấn luyện (dùng cho demo)
│   ├── naive_bayes.joblib
│   ├── softmax_regression.joblib
│   ├── mlp.pt
│   └── random_forest.joblib
├── results/                    # chỉ số đánh giá, kết quả khảo sát, cấu hình
├── figures/                    # biểu đồ, ma trận nhầm lẫn, hình so sánh
├── docs/
├── notebooks/
│   └── mnist_experiments_colab.ipynb   # đã chạy sẵn, có output
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── image_preprocessing.py
│   ├── demo_app.py             # ứng dụng demo vẽ & nhận dạng (Gradio)
│   └── models/
│       ├── naive_bayes.py
│       ├── softmax_regression.py
│       ├── mlp.py
│       └── ensemble.py
├── tests/
│   └── test_models.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 3. Kết quả đã chạy sẵn

| Vị trí | Nội dung |
|---|---|
| `notebooks/mnist_experiments_colab.ipynb` | Notebook đã chạy đầy đủ, giữ nguyên output; xem trực tiếp trên GitHub |
| `results/` | Chỉ số đánh giá, kết quả khảo sát và cấu hình |
| `figures/` | Biểu đồ, ma trận nhầm lẫn và hình so sánh |
| `artifacts/` | Mô hình đã huấn luyện (4 file) |

Các tệp tổng hợp gồm `final_metrics.csv`, `final_comparison.csv`, `sklearn_comparison.csv`, `confusion_pairs.csv` và `final_config.json`. Chi tiết thí nghiệm và phân tích kết quả được trình bày trong notebook và báo cáo đồ án.

## 4. Chạy ứng dụng demo (`src/demo_app.py`)

Ứng dụng web (Gradio) cho phép **vẽ một chữ số bằng chuột/cảm ứng** (hoặc tải ảnh lên) và xem dự đoán của cả bốn mô hình cùng lúc: chữ số dự đoán, độ tin cậy, phân bố xác suất và thời gian suy luận. Ảnh vẽ được chuẩn hoá giống MNIST (cắt vùng nét, co về 20×20, căn giữa theo tâm khối lượng trong khung 28×28).

> Demo chỉ **suy luận** bằng các model có sẵn trong `artifacts/`, không huấn luyện lại, và được thiết kế để chạy **cục bộ trên máy**.

**Bước 1 – Tải mã nguồn.**

```bash
git clone https://github.com/heleme69/MTH10322-PR-Midterm-Project.git
cd MTH10322-PR-Midterm-Project
```

**Bước 2 – Cài thư viện.** Nên dùng môi trường ảo (khuyến nghị Python 3.10 trở lên, vì Gradio và PyTorch bản mới yêu cầu):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` đã bao gồm đủ thư viện cho demo: `numpy`, `torch` (đọc `mlp.pt`), `scikit-learn` (nạp Random Forest, kéo theo `joblib`), `gradio` (giao diện) và `pillow` (xử lý ảnh), nên không cần cài thêm gì.

**Bước 3 – Chạy ứng dụng.** Từ **thư mục gốc repository**:

```bash
python -m src.demo_app
```

Terminal sẽ in `Đã nạp: Naive Bayes (...), Softmax Regression (...), MLP (...), Random Forest (...)` và tự mở trình duyệt tại **http://127.0.0.1:7860**. Vẽ một chữ số lên bảng, kết quả cập nhật ngay; hoặc mở mục *"Hoặc tải ảnh chữ số lên"* để nhận dạng ảnh có sẵn (tick *đảo màu* nếu ảnh nền đen, nét trắng).

**Tuỳ chọn dòng lệnh:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--artifacts <thư_mục>` | `artifacts/` ở thư mục gốc repo | Thư mục chứa model đã lưu |
| `--port <số>` | `7860` | Cổng chạy ứng dụng |
| `--share` | tắt | Tạo link công khai tạm thời qua Gradio |
| `--no-browser` | tắt | Không tự mở trình duyệt |

Ví dụ:

```bash
python -m src.demo_app --port 8000 --no-browser
```

**Lỗi thường gặp**

| Hiện tượng | Cách xử lý |
|---|---|
| `Thiếu file trong .../artifacts` | Thiếu file model; kiểm tra `artifacts/` có đủ 4 file (`naive_bayes.joblib`, `softmax_regression.joblib`, `mlp.pt`, `random_forest.joblib`) |
| `ModuleNotFoundError: gradio` / `torch` / `joblib` | Chạy lại `pip install -r requirements.txt` trong môi trường ảo đang bật |
| `ModuleNotFoundError: No module named 'src'` khi nạp model | Chạy từ **thư mục gốc repository** bằng `python -m src.demo_app` |
| Cổng 7860 đã bị dùng | Thêm `--port 7861` (hoặc cổng khác) |

### Chạy demo trên Google Colab (không cần cài Python)

Nếu không muốn cài gì trên máy, có thể chạy demo trong một notebook Colab mới. Dán lần lượt vào các cell:

```python
!git clone https://github.com/heleme69/MTH10322-PR-Midterm-Project.git
%cd MTH10322-PR-Midterm-Project
!pip install -q "gradio>=4.0" "pillow>=9.1"
!python -m src.demo_app --share --no-browser
```

Cell cuối chạy liên tục. Chờ dòng `Running on public URL: https://....gradio.live`, mở link đó trong trình duyệt để vẽ chữ số; bấm dừng cell khi dùng xong. Colab đã cài sẵn `numpy`, `torch` và `scikit-learn` nên chỉ cần cài thêm `gradio` và `pillow`, không cần cài toàn bộ `requirements.txt`.

## 5. Chạy notebook trên Google Colab (CPU)

Notebook đã được chạy sẵn (xem mục 3). Phần này dành cho việc chạy lại và huấn luyện lại từ đầu.

**Bước 1 – Tải mã nguồn và tạo ZIP.** Clone repository (xem mục 4, Bước 1), sau đó từ **thư mục gốc repository** tạo file `mnist_colab_source.zip` (các thư mục `src/`, `tests/`, `notebooks/` và tệp `requirements.txt` phải nằm trực tiếp trong ZIP).

**Windows (PowerShell hoặc CMD, Windows 10 trở lên):**

```powershell
tar -a -c -f mnist_colab_source.zip src tests notebooks requirements.txt
```

**Linux/macOS:**

```bash
zip -r mnist_colab_source.zip src tests notebooks requirements.txt
```

**Bước 2 – Upload ZIP.** Đặt `mnist_colab_source.zip` tại thư mục gốc **My Drive** của tài khoản Google dùng để chạy Colab. Notebook sẽ đọc:

```text
/content/drive/MyDrive/mnist_colab_source.zip
```

**Bước 3 – Chạy notebook.** Mở `notebooks/mnist_experiments_colab.ipynb` trên Google Colab, chọn **CPU** rồi chạy các cell theo thứ tự từ trên xuống. Notebook sẽ gắn Drive, giải nén mã nguồn, chuẩn bị dữ liệu, huấn luyện và đánh giá mô hình.

**Bước 4 – Xem kết quả.** Đầu ra được lưu trên Google Drive theo cấu trúc:

```text
MyDrive/MNIST_v1_3_outputs/v1_3_seed42_<SOURCE_FINGERPRINT>/
├── results/
├── figures/
└── artifacts/
```

Đường dẫn cụ thể của lần chạy được xác định bởi biến `OUTPUT_ROOT` trong notebook. Mỗi khi mã nguồn thay đổi, `<SOURCE_FINGERPRINT>` sẽ khác và notebook ghi vào thư mục mới, nên cần lấy kết quả từ thư mục của lần chạy mới nhất.

## 6. Kiểm thử và tái lập

Chạy unit test cục bộ, từ thư mục gốc repository (sau khi đã cài `requirements.txt`):

```bash
python -m pytest tests/test_models.py -v
```

Hoặc trong một cell Colab, sau khi notebook đã giải nén mã nguồn:

```python
!python -m pytest {PROJECT_ROOT}/tests/test_models.py -v
```

Bộ kiểm thử gồm **12 test**. Để huấn luyện lại thay vì sử dụng cache, đặt `USE_FIT_CACHE = False` trong notebook. Việc tái lập kết quả phụ thuộc vào mã nguồn, seed, dữ liệu và môi trường thực thi.

## 7. Tài nguyên

- **GitHub:** https://github.com/heleme69/MTH10322-PR-Midterm-Project
- **Kết quả thực nghiệm:** https://drive.google.com/drive/folders/1Z46gMqCoBIAX-HzQVAtT5JFGVxzu7_Qj
- **Biểu đồ và hình ảnh:** https://drive.google.com/drive/folders/1Lo55d6jmuT2Pg9kPYbrCsZW4vr8mclml
- **Bài thực hành tham khảo:** https://drive.google.com/drive/folders/1FZH8b6k4XHNkpL4tGAITW0_RWYzwieWV
