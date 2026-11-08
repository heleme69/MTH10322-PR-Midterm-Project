"""Web demo Gradio: vẽ chữ số, hiển thị dự đoán từ 4 mô hình đã train."""
from functools import lru_cache
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
from PIL import Image

from src.image_preprocessing import preprocess_canvas
from src.models.mlp import MLPFromScratch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Softmax Regression": "softmax_regression.joblib",
    "MLP": "mlp.pt",
    "Random Forest": "random_forest.joblib",
}


@lru_cache(maxsize=1)
def load_models():
    missing = [name for name in FILES.values() if not (ARTIFACTS / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Chưa có model artifacts: " + ", ".join(missing)
            + ". Hãy chạy: python -m scripts.train_and_export"
        )
    meta_path = ARTIFACTS / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    models = {}
    for name, filename in FILES.items():
        path = ARTIFACTS / filename
        if name == "MLP":
            model = MLPFromScratch(hidden_sizes=metadata.get("mlp", {}).get("hidden_sizes") or [128, 64])
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.model.load_state_dict(state)
            model.model.eval()
        else:
            # Chỉ nạp file .joblib do chính bạn huấn luyện/tạo ra.
            model = joblib.load(path)
        models[name] = model
    return models


def predict_canvas(editor_value):
    try:
        X_demo, preview = preprocess_canvas(editor_value)
        models = load_models()
        rows = []
        for name, model in models.items():
            prediction = int(model.predict(X_demo)[0])
            rows.append([name, prediction])
        summary = "### Kết quả nhận dạng\n" + "  \n".join(
            f"**{name}:** {digit}" for name, digit in rows
        )
        return preview, rows, summary
    except (ValueError, FileNotFoundError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc


def clear_all():
    return Image.new("RGB", (280, 280), "white"), None, [], "Chưa có dự đoán."


def training_notice():
    metadata_file = ARTIFACTS / "metadata.json"
    if not metadata_file.exists():
        return "**Chưa có mô hình đã lưu.** Hãy chạy script huấn luyện trước khi dự đoán."
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if any(info.get("limited_training") for info in metadata.values()):
        return "⚠️ **Chế độ thử nhanh:** Có mô hình chỉ được train trên một phần MNIST. Không dùng kết quả này làm số liệu báo cáo."
    return "Đã tìm thấy thông tin huấn luyện. Các mô hình sẽ được nạp khi dự đoán lần đầu."


def build_demo():
    with gr.Blocks(title="MNIST — Nhận dạng chữ số viết tay") as demo:
        gr.Markdown("# Nhận dạng chữ số viết tay (MNIST)\nVẽ một chữ số **0–9** rồi nhấn **Nhận dạng**.")
        gr.Markdown(training_notice())
        with gr.Row():
            with gr.Column():
                canvas = gr.Sketchpad(
                    value=Image.new("RGB", (280, 280), "white"),
                    label="Bảng vẽ (nét đen trên nền trắng)",
                    canvas_size=(280, 280), fixed_canvas=True,
                    type="numpy", image_mode="RGBA", height=350,
                    brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
                )
                with gr.Row():
                    btn = gr.Button("Nhận dạng", variant="primary")
                    clear = gr.Button("Xóa")
            with gr.Column():
                preview = gr.Image(label="Ảnh 28×28 mô hình thực sự nhận", image_mode="L",
                                   type="numpy", height=250, interactive=False)
                summary = gr.Markdown("Chưa có dự đoán.")
        results = gr.Dataframe(headers=["Mô hình", "Chữ số dự đoán"],
                               datatype=["str", "number"], interactive=False,
                               label="So sánh 4 mô hình")
        gr.Markdown("*Các mô hình được huấn luyện trước trên MNIST. App chỉ thực hiện suy luận, không huấn luyện lại.*")
        btn.click(fn=predict_canvas, inputs=[canvas], outputs=[preview, results, summary])
        clear.click(fn=clear_all, outputs=[canvas, preview, results, summary])
    return demo


if __name__ == "__main__":
    build_demo().launch(inbrowser=True)
