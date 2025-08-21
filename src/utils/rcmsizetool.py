import joblib
import numpy as np
import os

OUTPUT_DIR = "./src/model"

# Load model
# Lấy đường dẫn thư mục gốc project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Đường dẫn đến file model
MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "best_size_model_extended.joblib")
# Load model
bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
le = bundle["label_encoder_size"]
gmap = bundle["gender_map"]
feature_cols = bundle["feature_cols"]
canonical_order = bundle["canonical_order"]


def _normalize_gender(text: str) -> str:
    t = (text or "").strip().lower()
    if t in ["nam", "male", "m", "anh", "boy"]:
        return "Nam"
    if t in ["nữ", "nu", "female", "f", "chi", "girl"]:
        return "Nữ"
    # fallback: giữ nguyên nếu đã là "Nam"/"Nữ"
    return "Nam" if "nam" in t else ("Nữ" if "nữ" in t else "Nam")

def _fit_text_to_code(text: str) -> int:
    t = (text or "").strip().lower()
    if t in ["ôm", "om", "slim", "fitted", "tight"]:
        return 0
    if t in ["rộng", "rong", "loose", "oversize", "oversized", "baggy"]:
        return 2
    # mặc định "vừa"
    return 1

def shift_size_by_fit(base_size, fit_code):
    try:
        idx = canonical_order.index(base_size)
        if fit_code == 0 and idx > 0:
            return canonical_order[idx - 1]  # ôm hơn → xuống 1 size
        elif fit_code == 2 and idx < len(canonical_order) - 1:
            return canonical_order[idx + 1]  # rộng hơn → lên 1 size
        return base_size
    except:
        return base_size

def predict_size_with_fit(gender_text: str, height_cm: float, weight_kg: float,
                          fit_preference: int, apply_fit_rule: bool = True):
    g_key = _normalize_gender(gender_text)
    gender_code = gmap.get(gender_text.lower(), 0)
    X_new = np.array([[gender_code, height_cm, weight_kg, fit_preference]], dtype=float)
    y_pred = pipe.predict(X_new)[0]
    base_size = le.inverse_transform([y_pred])[0]
    final_size = shift_size_by_fit(base_size, fit_preference) if apply_fit_rule else base_size
    return base_size, final_size

# ---------- Wrapper public: nhận text từ chatbot ----------
def predict_size_public(gender_text: str, height_cm: float, weight_kg: float,
                        fit_text: str = "vừa", apply_fit_rule: bool = True):
    fit_code = _fit_text_to_code(fit_text)   # nhận cả "vừa/regular", "ôm/slim", "rộng/loose"
    return predict_size_with_fit(gender_text, height_cm, weight_kg, fit_code, apply_fit_rule)

if __name__ == "__main__":
    try:
        print("📏 Dự đoán Size Áo (Dùng mô hình mở rộng)")
        gender = input("Giới tính (Nam/Nữ): ").strip()
        height = float(input("Chiều cao (cm): "))
        weight = float(input("Cân nặng (kg): "))
        fit = input("Phong cách (ôm / vừa / rộng): ").strip().lower()
        
        fit_map = {"ôm": 0, "vừa": 1, "rộng": 2}
        if fit not in fit_map:
            raise ValueError("Phong cách phải là: ôm / vừa / rộng")
        fit_code = fit_map[fit]

        base, final = predict_size_with_fit(gender, height, weight, fit_code, apply_fit_rule=True)
        print(f"🎯 Kết quả: Size cơ bản: {base} → Sau điều chỉnh theo phong cách: {final}")

    except Exception as e:
        print("❌ Lỗi:", e)
