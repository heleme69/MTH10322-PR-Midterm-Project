from __future__ import annotations

import argparse
import base64
import html
import inspect
import io
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))  # để joblib nạp được src.models.*

# ===== Xử lý ảnh (chuẩn hoá giống MNIST) =====


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
    digit = Image.fromarray(np.ascontiguousarray(crop)).resize(
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


# ===== Giao diện =====



MODEL_COLORS = {
    "Naive Bayes": "#fbbf24",
    "Softmax Regression": "#22d3ee",
    "MLP": "#a78bfa",
    "Random Forest": "#34d399",
}
MODEL_TAGS = {
    "Naive Bayes": "Xác suất · Bayes",
    "Softmax Regression": "Tuyến tính · SGD",
    "MLP": "Mạng nơ-ron · Backprop",
    "Random Forest": "Ensemble · Bagging",
}

# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
CSS = r"""
:root{
  --bg:#0a0e1a; --panel:rgba(255,255,255,.045); --line:rgba(255,255,255,.09);
  --txt:#e8ecf6; --mut:#8a94ad; --acc1:#7c5cff; --acc2:#22d3ee;
  --ok:#34d399; --warn:#fbbf24; --bad:#fb7185;
}
html,body,.gradio-container,gradio-app{background:var(--bg)!important;color:var(--txt)!important;}
.gradio-container{
  max-width:1180px!important; margin:0 auto!important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif!important;
  background:
    radial-gradient(900px 480px at 8% -8%, rgba(124,92,255,.28), transparent 60%),
    radial-gradient(760px 420px at 100% 0%, rgba(34,211,238,.18), transparent 55%),
    var(--bg)!important;
}
footer,.built-with,.show-api{display:none!important;}
#canvas_b64{display:none!important;}
.main-row{gap:20px!important;align-items:flex-start!important;}
.gradio-container .block,.gradio-container .form{background:transparent!important;border:none!important;box-shadow:none!important;}
.gradio-container .prose{color:var(--txt)!important;}

/* Header */
.hdr{padding:34px 4px 22px;}
.hdr .eyebrow{display:inline-flex;gap:8px;align-items:center;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--acc2);font-weight:600;}
.hdr .eyebrow i{width:7px;height:7px;border-radius:50%;background:var(--acc2);box-shadow:0 0 12px var(--acc2);}
.hdr h1{margin:10px 0 8px;font-size:clamp(28px,4.4vw,44px);line-height:1.1;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(92deg,#fff 0%,#c7d2fe 45%,#67e8f9 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hdr p{margin:0;color:var(--mut);font-size:15px;max-width:720px;line-height:1.55;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.chip{font-size:12px;padding:6px 11px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:#c3cbe0;}

/* Panel (bảng vẽ) */
.pad{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:18px;
  backdrop-filter:blur(10px);box-shadow:0 20px 50px -25px rgba(0,0,0,.8);}
.pad-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;gap:10px;flex-wrap:wrap;}
.pad-title{font-weight:700;font-size:15px;}
.pad-hint{font-size:12px;color:var(--mut);}
.paper{position:relative;width:100%;max-width:340px;aspect-ratio:1/1;margin:0 auto;border-radius:16px;overflow:hidden;
  background:#fff;box-shadow:0 0 0 1px rgba(255,255,255,.14),0 0 0 6px rgba(124,92,255,.14),0 24px 60px -20px rgba(124,92,255,.55);
  transition:box-shadow .25s;}
.paper:hover{box-shadow:0 0 0 1px rgba(255,255,255,.22),0 0 0 6px rgba(34,211,238,.2),0 24px 60px -20px rgba(34,211,238,.5);}
#draw-canvas{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair;}
.paper .guide{position:absolute;inset:14.3%;border:1.5px dashed rgba(15,23,42,.13);border-radius:10px;pointer-events:none;}
.paper .empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:6px;color:rgba(15,23,42,.32);font-weight:600;pointer-events:none;transition:opacity .2s;font-size:15px;}
.paper .empty b{font-size:34px;line-height:1;opacity:.55}
.paper.has-ink .empty{opacity:0;}
.tools{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap;justify-content:center;}
.brush{display:flex;align-items:center;gap:9px;font-size:12px;color:var(--mut);flex:1 1 160px;}
.brush input[type=range]{flex:1;accent-color:#7c5cff;height:4px;}
.brush output{min-width:22px;text-align:right;color:var(--txt);font-variant-numeric:tabular-nums;}
.gradio-container .pad .btn{appearance:none;border:1px solid var(--line);background:rgba(255,255,255,.06);color:var(--txt);
  padding:8px 14px;border-radius:11px;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s;font-family:inherit;}
.gradio-container .pad .btn:hover{background:rgba(255,255,255,.12);transform:translateY(-1px);}
.gradio-container .pad .btn.danger{background:linear-gradient(135deg,rgba(251,113,133,.22),rgba(251,113,133,.1));border-color:rgba(251,113,133,.4);}
.gradio-container .pad .btn.danger:hover{background:rgba(251,113,133,.3);}
.kbd{margin-top:12px;text-align:center;font-size:11.5px;color:var(--mut);}
.kbd kbd{padding:1px 6px;border-radius:5px;border:1px solid var(--line);background:rgba(255,255,255,.05);font-family:inherit;color:#c3cbe0;}

/* Preview 28x28 */
.pv{display:flex;gap:16px;align-items:center;margin-top:14px;background:var(--panel);border:1px solid var(--line);
  border-radius:18px;padding:14px;}
.pv .img{width:92px;height:92px;flex:none;border-radius:10px;background:#000;border:1px solid var(--line);overflow:hidden;
  display:flex;align-items:center;justify-content:center;}
.pv img{width:100%;height:100%;image-rendering:pixelated;display:block;}
.pv .ph{color:#475569;font-size:11px;text-align:center;padding:6px;}
.pv h4{margin:0 0 4px;font-size:13.5px;}
.pv p{margin:0;font-size:12px;color:var(--mut);line-height:1.5;}

/* Kết quả */
.res{display:flex;flex-direction:column;gap:16px;}
.hero{display:flex;gap:22px;align-items:center;padding:22px 24px;border-radius:22px;border:1px solid var(--line);
  background:linear-gradient(135deg,rgba(124,92,255,.22),rgba(34,211,238,.1));position:relative;overflow:hidden;}
.hero::after{content:"";position:absolute;right:-60px;top:-60px;width:220px;height:220px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,255,255,.12),transparent 65%);}
.hero .digit{font-size:92px;line-height:1;font-weight:800;min-width:84px;text-align:center;letter-spacing:-.04em;
  background:linear-gradient(180deg,#fff,#a5b4fc);-webkit-background-clip:text;background-clip:text;color:transparent;
  filter:drop-shadow(0 8px 24px rgba(124,92,255,.55));animation:pop .35s ease;}
@keyframes pop{from{transform:scale(.8);opacity:.3}to{transform:scale(1);opacity:1}}
.hero .t{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);font-weight:600;}
.hero .h{font-size:20px;font-weight:700;margin:4px 0 8px;}
.badge{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:600;padding:5px 11px;border-radius:999px;
  border:1px solid;}
.badge.ok{color:var(--ok);border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.1);}
.badge.mid{color:var(--acc2);border-color:rgba(34,211,238,.4);background:rgba(34,211,238,.1);}
.badge.warn{color:var(--warn);border-color:rgba(251,191,36,.45);background:rgba(251,191,36,.1);}
.hero.idle .digit{background:none;color:#334155;filter:none;animation:none;-webkit-text-fill-color:#334155;}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}
@media(max-width:620px){.grid{grid-template-columns:1fr}.hero .digit{font-size:70px}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:15px 16px 12px;position:relative;
  transition:transform .15s,border-color .2s;}
.card:hover{transform:translateY(-2px);}
.card::before{content:"";position:absolute;left:16px;right:16px;top:0;height:2px;border-radius:0 0 4px 4px;background:var(--c);opacity:.85;}
.card.off{border-color:rgba(251,191,36,.55);background:rgba(251,191,36,.06);}
.card .hd{display:flex;align-items:center;gap:8px;}
.card .hd .dot{width:9px;height:9px;border-radius:50%;background:var(--c);box-shadow:0 0 10px var(--c);}
.card .hd b{font-size:14px;}
.card .hd small{margin-left:auto;font-size:11px;color:var(--mut);}
.card .row{display:flex;align-items:center;gap:14px;margin:12px 0 10px;}
.card .big{font-size:50px;line-height:1;font-weight:800;min-width:42px;text-align:center;color:var(--c);letter-spacing:-.03em;}
.card .cf{flex:1;min-width:0;}
.card .cf .n{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;}
.card .cf .l{font-size:11px;color:var(--mut);margin-bottom:6px;}
.bar{height:7px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden;}
.bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--c),#fff);transition:width .35s ease;}
.hist{display:grid;grid-template-columns:repeat(10,1fr);gap:4px;height:46px;align-items:end;margin-top:2px;}
.hist .c{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;gap:3px;}
.hist .c i{display:block;width:100%;border-radius:3px 3px 1px 1px;background:rgba(255,255,255,.16);min-height:2px;transition:height .35s ease;}
.hist .c.top i{background:var(--c);box-shadow:0 0 10px -1px var(--c);}
.hist .c span{font-size:9.5px;color:var(--mut);line-height:1;}
.hist .c.top span{color:var(--txt);font-weight:700;}
.card .ft{display:flex;justify-content:space-between;margin-top:9px;font-size:11px;color:var(--mut);}
.card.idle .big{color:#334155;} .card.idle .n{color:#475569;}
.note{font-size:12px;color:var(--mut);line-height:1.6;padding:2px 4px;}
.alert{padding:16px 18px;border-radius:16px;border:1px solid rgba(251,113,133,.45);background:rgba(251,113,133,.09);}
.alert b{display:block;margin-bottom:4px;color:#fecdd3;}
.alert span{font-size:13px;color:#fda4af;line-height:1.55;}
.alert.info{border-color:rgba(34,211,238,.4);background:rgba(34,211,238,.08);} .alert.info b{color:#a5f3fc}.alert.info span{color:#67e8f9}
"""

# ----------------------------------------------------------------------------
# HTML tĩnh
# ----------------------------------------------------------------------------
HEADER_HTML = """
<div class="hdr">
  <div class="eyebrow"><i></i>Nhận dạng mẫu · MNIST</div>
  <h1>Vẽ một chữ số,<br>xem dự đoán của 4 mô hình</h1>
</div>
"""

PAD_HTML = """
<div class="pad">
  <div class="pad-top"><span class="pad-title">Bảng vẽ</span>
    <span class="pad-hint">Nét đen trên nền trắng · vẽ to, nằm giữa khung nét đứt</span></div>
  <div class="paper" id="paper">
    <canvas id="draw-canvas" width="280" height="280"></canvas>
    <div class="guide"></div>
    <div class="empty"><b>✍︎</b>Vẽ chữ số ở đây</div>
  </div>
  <div class="tools">
    <label class="brush">Nét
      <input id="brush-size" type="range" min="12" max="34" value="22"><output id="brush-out">22</output></label>
    <button type="button" class="btn" id="btn-undo">↶ Hoàn tác</button>
    <button type="button" class="btn danger" id="btn-clear">✕ Xoá</button>
  </div>
  <div class="kbd"><kbd>Ctrl</kbd>+<kbd>Z</kbd> hoàn tác · <kbd>Esc</kbd> xoá · hỗ trợ cảm ứng &amp; bút</div>
</div>
"""

# ----------------------------------------------------------------------------
# JS: vẽ trên canvas, gửi PNG sang Textbox ẩn (cầu nối tới Python)
# ----------------------------------------------------------------------------
PAD_JS = r"""
(function () {
  var BRIDGE = '#canvas_b64';

  function send(val) {
    var el = document.querySelector(BRIDGE + ' textarea, ' + BRIDGE + ' input');
    if (!el) { return false; }
    var proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, val);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  }

  function init(canvas) {
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    var paper = document.getElementById('paper');
    var slider = document.getElementById('brush-size');
    var out = document.getElementById('brush-out');
    var W = canvas.width, H = canvas.height;
    var drawing = false, last = null, prevMid = null, timer = null;
    var hasInk = false, stack = [];

    function blank() { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H); }
    function style() {
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.strokeStyle = '#000'; ctx.fillStyle = '#000';
      ctx.lineWidth = +slider.value;
    }
    function mark() { paper.classList.toggle('has-ink', hasInk); }
    function snapshot() {
      stack.push({ img: ctx.getImageData(0, 0, W, H), ink: hasInk });
      if (stack.length > 40) { stack.shift(); }
    }
    function push() {
      if (timer) { clearTimeout(timer); timer = null; }
      send(Date.now() + '|' + (hasInk ? canvas.toDataURL('image/png') : ''));
    }
    function pos(e) {
      var r = canvas.getBoundingClientRect();
      return { x: (e.clientX - r.left) * W / r.width, y: (e.clientY - r.top) * H / r.height };
    }

    blank(); style(); mark();

    canvas.addEventListener('pointerdown', function (e) {
      e.preventDefault();
      canvas.setPointerCapture(e.pointerId);
      snapshot(); style();
      drawing = true; hasInk = true; mark();
      var p = pos(e); last = p; prevMid = p;
      ctx.beginPath(); ctx.arc(p.x, p.y, ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill();
    });
    canvas.addEventListener('pointermove', function (e) {
      if (!drawing) { return; }
      var evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      if (!evs.length) { evs = [e]; }
      for (var i = 0; i < evs.length; i++) {
        var p = pos(evs[i]);
        var mid = { x: (last.x + p.x) / 2, y: (last.y + p.y) / 2 };
        ctx.beginPath(); ctx.moveTo(prevMid.x, prevMid.y);
        ctx.quadraticCurveTo(last.x, last.y, mid.x, mid.y); ctx.stroke();
        last = p; prevMid = mid;
      }
      if (!timer) { timer = setTimeout(push, 350); }   // dự đoán trực tiếp khi đang vẽ
    });
    function end() {
      if (!drawing) { return; }
      drawing = false;
      setTimeout(push, 60);
    }
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);

    slider.addEventListener('input', function () { out.textContent = slider.value; style(); });

    function undo() {
      var s = stack.pop();
      if (!s) { return; }
      ctx.putImageData(s.img, 0, 0); hasInk = s.ink; style(); mark(); push();
    }
    function clear() {
      snapshot(); blank(); hasInk = false; style(); mark(); push();
    }
    document.getElementById('btn-undo').addEventListener('click', undo);
    document.getElementById('btn-clear').addEventListener('click', clear);
    document.addEventListener('keydown', function (e) {
      var tag = (document.activeElement && document.activeElement.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA') { return; }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
      else if (e.key === 'Escape') { clear(); }
    });
    canvas.dataset.ready = '1';
  }

  function tryInit() {
    var c = document.getElementById('draw-canvas');
    if (c && !c.dataset.bound) { c.dataset.bound = '1'; init(c); }
  }
  new MutationObserver(tryInit).observe(document.documentElement, { childList: true, subtree: true });
  tryInit();
})();
"""

HEAD = "<script>" + PAD_JS + "</script>"


# ----------------------------------------------------------------------------
# Python helpers
# ----------------------------------------------------------------------------
def decode_payload(payload: str | None) -> Image.Image | None:
    """'timestamp|data:image/png;base64,....' -> PIL Image (None nếu bảng trống)."""
    if not payload:
        return None
    _, _, data = payload.partition("|")
    if "," in data:
        data = data.split(",", 1)[1]
    if not data.strip():
        return None
    return Image.open(io.BytesIO(base64.b64decode(data)))


def normalize_upload(img: Image.Image, invert: bool = False) -> Image.Image:
    """Ảnh tải lên -> ảnh xám nét đen/nền trắng, đúng quy ước của preprocess_canvas."""
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(bg, rgba).convert("L")
    return ImageOps.invert(gray) if invert else gray


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8))).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_preview(preview28: np.ndarray | None) -> str:
    if preview28 is None:
        img = '<div class="ph">28×28<br>chờ nét vẽ</div>'
    else:
        img = f'<img alt="28x28" src="data:image/png;base64,{_png_b64(preview28)}">'
    return f"""
<div class="pv"><div class="img">{img}</div>
  <div><h4>Ảnh 28×28 mà mô hình thực sự nhận</h4>
  <p>Chữ số được cắt sát nét, co về khung 20×20 rồi dời tâm khối lượng về giữa ảnh — đúng cách MNIST được tạo ra.</p></div>
</div>"""


def render_notice(title: str, message: str, kind: str = "error") -> str:
    cls = "alert" if kind == "error" else "alert info"
    return (f'<div class="{cls}"><b>{html.escape(title)}</b>'
            f'<span>{html.escape(message)}</span></div>')


def _hist(probs, top, color_var=True) -> str:
    cols = []
    for d, p in enumerate(probs):
        h = max(2, round(p * 100))
        cols.append(f'<div class="c{" top" if d == top else ""}"><i style="height:{h}%"></i><span>{d}</span></div>')
    return '<div class="hist">' + "".join(cols) + "</div>"


def _card(name: str, info: str, r=None, consensus: int | None = None) -> str:
    color = MODEL_COLORS.get(name, "#a78bfa")
    tag = html.escape(info or MODEL_TAGS.get(name, ""))
    if r is None:  # trạng thái chờ
        return f"""
<div class="card idle" style="--c:{color}">
  <div class="hd"><span class="dot"></span><b>{html.escape(name)}</b><small>{tag}</small></div>
  <div class="row"><div class="big">–</div><div class="cf"><div class="n">––.–%</div>
    <div class="l">độ tin cậy</div><div class="bar"><i style="width:0%"></i></div></div></div>
  {_hist([0] * 10, -1)}
  <div class="ft"><span>{html.escape(MODEL_TAGS.get(name, ''))}</span><span>chờ nét vẽ</span></div>
</div>"""
    off = " off" if consensus is not None and r.digit != consensus else ""
    return f"""
<div class="card{off}" style="--c:{color}">
  <div class="hd"><span class="dot"></span><b>{html.escape(name)}</b><small>{tag}</small></div>
  <div class="row"><div class="big">{r.digit}</div><div class="cf">
    <div class="n">{r.confidence * 100:.1f}%</div><div class="l">độ tin cậy cho chữ số {r.digit}</div>
    <div class="bar"><i style="width:{r.confidence * 100:.1f}%"></i></div></div></div>
  {_hist(r.probs, r.digit)}
  <div class="ft"><span>{html.escape(MODEL_TAGS.get(name, ''))}</span><span>⏱ {r.ms:.2f} ms</span></div>
</div>"""


def render_results(pred=None, info: dict | None = None, names: list[str] | None = None) -> str:
    info = info or {}
    if pred is None:
        names = names or list(MODEL_COLORS)
        cards = "".join(_card(n, info.get(n, "")) for n in names)
        return f"""
<div class="res">
  <div class="hero idle"><div class="digit">?</div><div>
    <div class="t">Kết quả đồng thuận</div><div class="h">Hãy vẽ một chữ số 0–9</div>
    <span class="badge mid">Bốn mô hình sẽ bỏ phiếu ngay khi bạn nhấc bút</span></div></div>
  <div class="grid">{cards}</div></div>"""

    digit, votes = pred.consensus
    n = len(pred.results)
    agree = [r for r in pred.results if r.digit == digit]
    mean_conf = float(np.mean([r.confidence for r in agree])) * 100
    if votes == n:
        title, cls, txt = "Cả bốn mô hình đồng ý", "ok", f"{votes}/{n} phiếu · tin cậy TB {mean_conf:.1f}%"
    elif votes > n / 2:
        title, cls, txt = "Đa số đồng ý", "mid", f"{votes}/{n} phiếu · có mô hình bất đồng"
    else:
        title, cls, txt = "Các mô hình chưa thống nhất", "warn", f"chỉ {votes}/{n} phiếu cho chữ số {digit} — hãy vẽ rõ hơn"
    cards = "".join(_card(r.name, info.get(r.name, ""), r, digit) for r in pred.results)
    return f"""
<div class="res">
  <div class="hero"><div class="digit">{digit}</div><div>
    <div class="t">Kết quả đồng thuận</div><div class="h">{title}</div>
    <span class="badge {cls}">{txt}</span></div></div>
  <div class="grid">{cards}</div>
  <div class="note">Viền vàng = mô hình bất đồng với số phiếu đa số. “Độ tin cậy” là xác suất đầu ra
    (softmax / log-posterior / tỉ lệ phiếu của các cây) nên chưa chắc đã được hiệu chuẩn — Naive Bayes và
    Softmax thường tự tin quá mức.</div>
</div>"""


# ===== Nạp mô hình & suy luận =====



ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS = ROOT / "artifacts"

MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Softmax Regression": "softmax_regression.joblib",
    "MLP": "mlp.pt",
    "Random Forest": "random_forest.joblib",
}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ----------------------------------------------------------------------------
# Bộ chuyển đổi: model đã huấn luyện -> hàm proba(X) trả về (n, 10)
# ----------------------------------------------------------------------------
def _nb_proba(model, X):
    cols = []
    for c in model.classes_:
        cols.append(np.log(model.priors_[c])
                    + model.log_pdf(X, model.means_[c], None, alpha=model.alpha))
    return _softmax(np.array(cols).T)


def _softmax_reg_proba(model, X):
    return _softmax(X @ model.W + model.b)


def _rf_proba(model, X):
    votes = np.array([t.predict(X.astype(np.float32)) for t in model.trees])  # (T, n)
    n_classes = 10
    out = np.zeros((X.shape[0], n_classes))
    for j in range(X.shape[0]):
        out[j] = np.bincount(votes[:, j].astype(int), minlength=n_classes)[:n_classes]
    return out / len(model.trees)


class NumpyMLP:
    """Suy luận MLP bằng NumPy từ state_dict (Linear-ReLU-...-Linear).

    Dropout chỉ có tác dụng khi train nên bị bỏ qua; các lớp Linear được nhận
    diện qua khoá '*.weight' / '*.bias' theo thứ tự chỉ số lớp.
    """

    def __init__(self, state_dict):
        items = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                               dtype=np.float32)
                 for k, v in state_dict.items()}
        idx = sorted({int(k.split(".")[0]) for k in items if k.endswith(".weight")})
        self.layers = [(items[f"{i}.weight"], items[f"{i}.bias"]) for i in idx]
        self.sizes = [self.layers[0][0].shape[1]] + [w.shape[0] for w, _ in self.layers]

    def logits(self, X):
        a = np.asarray(X, dtype=np.float32)
        for k, (W, b) in enumerate(self.layers):
            a = a @ W.T + b
            if k < len(self.layers) - 1:
                a = np.maximum(a, 0.0)
        return a

    def proba(self, X):
        return _softmax(self.logits(X))


# ----------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    digit: int
    confidence: float
    probs: list[float]
    ms: float


@dataclass
class Prediction:
    results: list[ModelResult] = field(default_factory=list)

    @property
    def consensus(self) -> tuple[int, int]:
        """(chữ số được nhiều mô hình chọn nhất, số mô hình đồng ý)."""
        votes = np.bincount([r.digit for r in self.results], minlength=10)
        return int(votes.argmax()), int(votes.max())


class ModelHub:
    """Nạp 4 mô hình một lần và dự đoán cho ảnh (1, 784) float32 trong [0, 1]."""

    def __init__(self, artifacts_dir: Path | str = DEFAULT_ARTIFACTS):
        self.dir = Path(artifacts_dir)
        self._proba = {}
        self.info = {}
        self._load()

    def _load(self):
        missing = [f for f in MODEL_FILES.values() if not (self.dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Thiếu file trong {self.dir}: {', '.join(missing)}. "
                "Hãy chạy Notebook đến mục 10.7 'Lưu model cuối' rồi chép thư mục "
                "artifacts/ về cạnh src/."
            )
        import joblib  # chỉ nạp file .joblib do chính bạn huấn luyện

        nb = joblib.load(self.dir / "naive_bayes.joblib")
        sm = joblib.load(self.dir / "softmax_regression.joblib")
        rf = joblib.load(self.dir / "random_forest.joblib")
        self._proba["Naive Bayes"] = lambda X, m=nb: _nb_proba(m, X)
        self._proba["Softmax Regression"] = lambda X, m=sm: _softmax_reg_proba(m, X)
        self._proba["Random Forest"] = lambda X, m=rf: _rf_proba(m, X)
        self.info["Random Forest"] = f"{len(rf.trees)} cây · depth ≤ {rf.max_depth}"
        self.info["Softmax Regression"] = f"W {tuple(sm.W.shape)}"
        self.info["Naive Bayes"] = f"Multinomial · α = {nb.alpha}"

        import torch
        blob = torch.load(self.dir / "mlp.pt", map_location="cpu", weights_only=True)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        mlp = NumpyMLP(state)
        self._proba["MLP"] = mlp.proba
        self.info["MLP"] = " → ".join(str(s) for s in mlp.sizes)

        # Giữ đúng thứ tự hiển thị
        self._proba = {k: self._proba[k] for k in MODEL_FILES}

    @property
    def names(self):
        return list(self._proba)

    def predict(self, X: np.ndarray) -> Prediction:
        X = np.asarray(X, dtype=np.float32).reshape(1, 784)
        out = Prediction()
        for name, fn in self._proba.items():
            t0 = time.perf_counter()
            p = np.asarray(fn(X))[0]
            ms = (time.perf_counter() - t0) * 1000
            out.results.append(ModelResult(
                name=name, digit=int(p.argmax()), confidence=float(p.max()),
                probs=[float(v) for v in p], ms=ms))
        return out


# ===== Ứng dụng Gradio =====




_ARTIFACTS = DEFAULT_ARTIFACTS


@lru_cache(maxsize=2)
def get_hub(artifacts_dir: str) -> ModelHub:
    """Nạp mô hình một lần (lỗi không bị cache nên sửa xong có thể thử lại)."""
    return ModelHub(artifacts_dir)


def _run(image):
    """PIL (nét đen/nền trắng) -> (html_preview, html_kết_quả)."""
    try:
        X, preview = preprocess_canvas(image)
        hub = get_hub(str(_ARTIFACTS))
        pred = hub.predict(X)
        return render_preview(preview), render_results(pred, hub.info, hub.names)
    except ValueError as exc:       # bảng trống / ảnh không có nét
        return render_preview(None), render_notice("Chưa có nét vẽ", str(exc), "info")
    except FileNotFoundError as exc:  # chưa có artifacts
        return render_preview(None), render_notice("Chưa tìm thấy mô hình đã huấn luyện", str(exc))
    except Exception as exc:        # không để app sập vì một lỗi bất ngờ
        return render_preview(None), render_notice(f"Lỗi {type(exc).__name__}", str(exc))


def on_canvas(payload):
    image = decode_payload(payload)
    if image is None:
        return render_preview(None), render_results(None)
    return _run(image)


def on_upload(img, invert):
    if img is None:
        return render_preview(None), render_results(None)
    return _run(normalize_upload(img, bool(invert)))


def _style_kwargs():
    import gradio as gr
    """css/head/theme nằm ở Blocks() (Gradio 4–5) hoặc launch() (Gradio 6+)."""
    try:
        theme = gr.themes.Base(primary_hue="violet", neutral_hue="slate")
    except Exception:
        theme = None
    wanted = {"css": CSS, "head": HEAD, "theme": theme}
    wanted = {k: v for k, v in wanted.items() if v is not None}
    accepted = set(inspect.signature(gr.Blocks.__init__).parameters)
    return ({k: v for k, v in wanted.items() if k in accepted},
            {k: v for k, v in wanted.items() if k not in accepted})


def build_demo():
    import gradio as gr
    blocks_kw, launch_kw = _style_kwargs()
    with gr.Blocks(title="MNIST · Vẽ và nhận dạng chữ số", **blocks_kw) as demo:
        gr.HTML(HEADER_HTML)
        with gr.Row(elem_classes="main-row"):
            with gr.Column(scale=5, min_width=320):
                gr.HTML(PAD_HTML)
                preview = gr.HTML(render_preview(None))
                with gr.Accordion("Hoặc tải ảnh chữ số lên", open=False):
                    up = gr.Image(type="pil", label="Ảnh chữ số", image_mode="RGBA")
                    invert = gr.Checkbox(label="Ảnh nền đen, nét trắng (đảo màu)", value=False)
                    go = gr.Button("Nhận dạng ảnh này", variant="primary")
            with gr.Column(scale=7, min_width=420):
                results = gr.HTML(render_results(None))

        # Textbox ẩn: JS của bảng vẽ ghi PNG base64 vào đây để chuyển sang Python.
        bridge = gr.Textbox(elem_id="canvas_b64", show_label=False, container=False)
        bridge.change(on_canvas, inputs=bridge, outputs=[preview, results],
                      show_progress="hidden", trigger_mode="always_last")
        go.click(on_upload, inputs=[up, invert], outputs=[preview, results], show_progress="hidden")
    return demo, launch_kw


def main():
    global _ARTIFACTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS), help="thư mục chứa model đã lưu")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="tạo link công khai (Gradio share)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    _ARTIFACTS = Path(args.artifacts)

    try:  # nạp sớm để báo lỗi rõ ngay ở terminal; app vẫn mở được và hiện hướng dẫn
        hub = get_hub(str(_ARTIFACTS))
        print("Đã nạp:", ", ".join(f"{n} ({hub.info.get(n, '')})" for n in hub.names))
    except Exception as exc:
        print("CẢNH BÁO:", exc)

    demo, launch_kw = build_demo()
    demo.queue().launch(server_port=args.port, share=args.share,
                        inbrowser=not args.no_browser, **launch_kw)


if __name__ == "__main__":
    main()
