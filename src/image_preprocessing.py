"""Chuyển ảnh từ Gradio Sketchpad thành input MNIST: (1, 784), float32, [0,1]."""
import numpy as np
from PIL import Image


def _canvas_to_ink(editor_value):
    """Nhận dict ImageEditor/Sketchpad hoặc ảnh; trả về mảng độ đậm nét vẽ [0,255]."""
    if editor_value is None:
        raise ValueError("Bạn chưa vẽ chữ số nào.")
    if isinstance(editor_value, dict):
        image = editor_value.get("composite")
        if image is None:
            raise ValueError("Không đọc được ảnh từ bảng vẽ.")
    else:
        image = editor_value

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    elif not isinstance(image, Image.Image):
        raise TypeError("Ảnh đầu vào phải là NumPy array hoặc PIL Image.")

    # Nếu nền trong suốt, ghép lên nền trắng để nét đen vẫn được nhận diện.
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white, rgba).convert("L")
    # Canvas: nét đen / nền trắng; MNIST: nét sáng / nền đen.
    return 255 - np.asarray(composite, dtype=np.uint8)


def preprocess_canvas(editor_value):
    """Trả về (X_demo, preview): X_demo (1,784) float32; preview (28,28) uint8."""
    ink = _canvas_to_ink(editor_value)
    foreground = ink > 25  # bỏ nhiễu rất mờ
    if not np.any(foreground):
        raise ValueError("Bảng vẽ đang trống. Hãy vẽ một chữ số 0–9.")

    rows, cols = np.where(foreground)
    # Lấy vùng nét vẽ, không thu nhỏ cả vùng Canvas trống.
    crop = ink[rows.min():rows.max()+1, cols.min():cols.max()+1]
    h, w = crop.shape
    ratio = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    digit = Image.fromarray(crop, mode="L").resize(
        (new_w, new_h), Image.Resampling.LANCZOS
    )

    # Đặt bounding box vào giữa ảnh 28x28, giữ tỷ lệ gốc.
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28-new_w)//2, (28-new_h)//2))
    # MNIST đặt chữ số gần tâm khối lượng pixel, không chỉ giữa bounding box.
    arr = np.asarray(canvas, dtype=np.float64)
    total = arr.sum()
    if total > 0:
        yy, xx = np.indices(arr.shape)
        dx = int(np.rint(13.5 - (xx * arr).sum() / total))
        dy = int(np.rint(13.5 - (yy * arr).sum() / total))
        shifted = np.zeros_like(arr, dtype=np.uint8)
        sy0, sy1 = max(0, -dy), min(28, 28-dy)
        sx0, sx1 = max(0, -dx), min(28, 28-dx)
        shifted[sy0+dy:sy1+dy, sx0+dx:sx1+dx] = arr[sy0:sy1, sx0:sx1].astype(np.uint8)
        preview = shifted
    else:
        preview = np.asarray(canvas, dtype=np.uint8).copy()
    X_demo = (preview.astype(np.float32) / 255.0).reshape(1, 784)
    return X_demo, preview
