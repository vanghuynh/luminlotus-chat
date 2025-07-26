import os
import pandas as pd
import joblib
from catboost import CatBoostClassifier
import os

# Load model và encoder
current_dir = os.path.dirname(os.path.abspath(__file__))  # Thư mục chứa file rcmsizetool.py
model_path = os.path.join(current_dir, "../model/catboost_pipeline_model_final.cbm")
encoder_path = os.path.join(current_dir, "../model/catboost_pipeline_label_encoder_final.pkl")
model = CatBoostClassifier()
model.load_model(model_path)
le = joblib.load(encoder_path)

# Hàm dự đoán size với giá trị mặc định cho các trường phụ
def predict_size(height, weight, gender, age=None, length_back=None, chest=None):
    bmi = weight / ((height / 100) ** 2)

    # Gán giá trị mặc định nếu không có
    if age is None:
        age = 25
    if length_back is None:
        length_back = 72.0
    if chest is None:
        chest = 100.0

    input_df = pd.DataFrame([[height, weight, age, gender.lower(), bmi, length_back, chest]],
                             columns=['height', 'weight', 'age', 'gender', 'BMI', 'length_back', 'chest'])
    
    pred_encoded = model.predict(input_df)[0]
    predicted_size = le.inverse_transform([int(pred_encoded)])[0]
    return {"recommended_size": predicted_size}

# Chạy local bằng giao diện dòng lệnh
if __name__ == "__main__":
    try:
        height = float(input("Nhập chiều cao (cm): "))
        weight = float(input("Nhập cân nặng (kg): "))
        gender = input("Nhập giới tính (Nam/Nữ): ").strip().lower()

        if gender not in ['nam', 'nữ', 'male', 'female']:
            print("⚠️ Giới tính không hợp lệ. Chỉ chấp nhận: Nam/Nữ hoặc male/female.")
        else:
            # Các input tùy chọn
            age_input = input("Nhập tuổi (tùy chọn, Enter nếu bỏ qua): ").strip()
            length_back_input = input("Nhập chiều dài lưng (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            chest_input = input("Nhập vòng ngực (cm, tùy chọn, Enter nếu bỏ qua): ").strip()

            # Chuyển đổi nếu có
            age = int(age_input) if age_input else None
            length_back = float(length_back_input) if length_back_input else None
            chest = float(chest_input) if chest_input else None

            # Dự đoán
            result = predict_size(height, weight, gender, age, length_back, chest)
            print(f"\n🎯 Recommended size: {result}")
    except Exception as e:
        print("❌ Lỗi khi nhập hoặc xử lý dữ liệu:", e)
