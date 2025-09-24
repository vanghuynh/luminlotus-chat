import joblib
import numpy as np
import os
import re
import pandas as pd

# ==== ĐƯỜNG DẪN MODEL ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "best_model_rf.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "src", "model", "label_encoder_size.pkl")

# ==== LOAD MODEL & LABEL ENCODER ====
pipe = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Các size theo thứ tự logic
canonical_order = ["XXS", "S", "M", "L", "XL", "XXL", "XXXL"]

# Map phong cách
fit_map = {"ôm": -1, "vừa": 0, "rộng": 1}


# ====== RÀNG BUỘC ======
def validate_inputs(height: float, weight: float):
    if not (130 <= height <= 200):
        raise ValueError("⚠️Không có size phù hợp cho bạn.")
    if not (30 <= weight <= 150):
        raise ValueError("⚠️Không có size phù hợp cho bạn.")


# ====== CHUẨN HÓA ĐẦU VÀO ======
def _normalize_gender(text: str) -> str:
    t = (text or "").strip().lower()
    if t in ["nam", "male", "m", "anh", "boy"]:
        return "Nam"
    if t in ["nữ", "nu", "female", "f", "chi", "girl"]:
        return "Nữ"
    return "Nam" if "nam" in t else ("Nữ" if "nữ" in t else "Nam")

def _fit_text_to_code(text: str) -> int:
    t = (text or "").strip().lower()
    if t in ["ôm", "om", "slim", "fitted", "tight"]:
        return -1
    if t in ["rộng", "rong", "loose", "oversize", "oversized", "baggy"]:
        return 1
    return 0   # mặc định "vừa"

def _parse_height(text: str):
    t = text.lower().replace(",", ".").strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(\d+)?", t)
    if m:
        meters = float(m.group(1))
        extra = m.group(2)
        if extra:
            return meters * 100 + float(extra)
        return meters * 100
    cm = re.search(r"(\d+(?:\.\d+)?)\s*cm", t)
    if cm:
        return float(cm.group(1))
    only = re.search(r"\b(\d+(?:\.\d+)?)\b", t)
    if only:
        v = float(only.group(1))
        if v >= 100: return v
        if v < 3:   return v * 100
    return None

def _parse_weight(text: str):
    t = text.lower().replace(",", ".").strip()
    w = re.search(r"(\d+(?:\.\d+)?)(?:\s*(kg|kgs|ký|kí|kilo|kilogram))", t)
    if w:
        return float(w.group(1))
    g = re.search(r"(\d+(?:\.\d+)?)\s*g(ram)?", t)
    if g:
        return float(g.group(1)) / 1000
    only = re.search(r"\b(\d+(?:\.\d+)?)\b", t)
    if only:
        return float(only.group(1))
    return None

def _parse_fit(text: str):
    t = text.lower()
    if any(k in t for k in ["ôm", "om", "slim", "fitted", "tight"]):
        return "ôm"
    if any(k in t for k in ["rộng", "rong", "loose", "oversize", "oversized", "baggy"]):
        return "rộng"
    return "vừa"


# ====== DỰ ĐOÁN SIZE ======
def shift_size(base_size, fit_code):
    try:
        idx = canonical_order.index(base_size)
        new_idx = idx + fit_code
        if 0 <= new_idx < len(canonical_order):
            return canonical_order[new_idx]
        return base_size
    except:
        return base_size

def predict_size(height, weight, gender, fit_style="vừa"):
    validate_inputs(height, weight)
    gender_code = 0 if _normalize_gender(gender) == "Nữ" else 1
    fit_pref = fit_map.get(fit_style, 0)

    X = pd.DataFrame([{
        "gender_code": gender_code,
        "height_cm": height,
        "weight_kg": weight,
        "fit_preference": fit_pref
    }])

    base_label = pipe.predict(X)[0]
    base_size = le.inverse_transform([base_label])[0]
    final_size = shift_size(base_size, fit_pref)
    return base_size, final_size


# ====== HÀM PUBLIC: NHẬN TEXT TỰ DO ======
def predict_size_public_text(user_text: str) -> str:
    height = _parse_height(user_text)
    weight = _parse_weight(user_text)
    gender = _normalize_gender(user_text)
    fit = _parse_fit(user_text)

    if height is None or weight is None:
        return "⚠️ Vui lòng cung cấp chiều cao (cm) và cân nặng (kg)."

    base, final = predict_size(height, weight, gender, fit)
    return f"🎯 Size cơ bản: {base}. ✨ Theo phong cách '{fit}': {final}."


#________________Test nhận input từ local_________________
if __name__ == "__main__":
    try:
        print("📏 Dự đoán Size Áo (Dùng mô hình mở rộng)")
        gender = input("Giới tính (Nam/Nữ): ").strip()
        height = float(input("Chiều cao (cm): "))
        weight = float(input("Cân nặng (kg): "))
        fit = input("Phong cách (ôm / vừa / rộng): ").strip().lower()

        # gọi trực tiếp hàm dự đoán
        base, final = predict_size(height, weight, gender, fit)

        print(f"Giới tính: {gender}")
        print(f"Chiều cao: {height} cm | Cân nặng: {weight} kg")
        print(f"Size cơ bản: {base}")
        print(f"Sau điều chỉnh phong cách ({fit}): {final}")

    except Exception as e:
        print("Lỗi:", e)