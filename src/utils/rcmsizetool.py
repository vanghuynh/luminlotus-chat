import pandas as pd
import joblib
import numpy as np
import os


# Xác định đường dẫn tuyệt đối đến thư mục chứa model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "model", "rf_full_pipeline.pkl")
encoder_path = os.path.join(current_dir, "..", "model", "rf_full_label_encoder.pkl")

# Load mô hình và label encoder
pipeline = joblib.load(os.path.abspath(model_path))
label_encoder = joblib.load(os.path.abspath(encoder_path))

size_description = {
    'nữ': {
        'XS': {"target_audience": "Vóc dáng nhỏ nhắn, mảnh mai, muốn áo ôm gọn.", "style": "Ôm vừa phải, tôn dáng gọn gàng, năng động."},
        'S': {"target_audience": "Thích mặc áo rộng rãi, thoải mái hơn size XS, hoặc dáng người vừa phải.", "style": "Trẻ trung, thoải mái, dễ phối đồ."},
        'M': {"target_audience": "Phù hợp với đa số nữ giới muốn sự thoải mái.", "style": "Rộng vừa phải, thoải mái tối đa, che khuyết điểm nhẹ."},
        'L': {"target_audience": "Thích mặc rộng rãi, thoải mái hoặc có vóc dáng đầy đặn hơn.", "style": "Tạo sự thoải mái tối đa, phóng khoáng, có thể mặc dáng oversized nhẹ."},
        'XL': {"target_audience": "Rất cao, hoặc muốn mặc áo thật rộng rãi, phom phóng khoáng.", "style": "Thoải mái tối đa, phom dáng rộng, phù hợp phong cách cá tính."},
        '2XL': {"target_audience": "Vóc dáng lớn hoặc rất cao, muốn mặc áo siêu rộng (oversized).", "style": "Thoải mái vượt trội, phong cách độc đáo, ấn tượng."}
    },
    'nam': {
        'XS': {"target_audience": "Rất gầy, thích mặc áo ôm sát người.", "style": "Ôm sát, tôn lên vóc dáng gọn gàng."},
        'S': {"target_audience": "Vóc dáng vừa phải, thích mặc áo ôm vừa vặn.", "style": "Lịch sự, gọn gàng, năng động."},
        'M': {"target_audience": "Phù hợp với đa số nam giới, muốn áo vừa vặn, thoải mái.", "style": "Vừa vặn, không quá ôm cũng không quá rộng, năng động."},
        'L': {"target_audience": "Vóc dáng trung bình khá đến đầy đặn, muốn áo rộng rãi.", "style": "Thoải mái, phóng khoáng, dễ vận động."},
        'XL': {"target_audience": "Vóc dáng cao to.", "style": "Rộng rãi, thoải mái tối đa, phù hợp phong cách Streetwear."},
        '2XL': {"target_audience": "Vóc dáng lớn, cao trên 1m80 và/hoặc cân nặng trên 100kg.", "style": "Thoải mái vượt trội, phong cách thể thao hoặc cá tính mạnh mẽ."}
    }
}

# Giá trị trung bình của các feature còn lại theo giới tính
mean_values = {
    'nam': {
        'chest': 100,
        'waist': 100,
        'shoulder': 50
    },
    'nữ': {
        'chest': 110,
        'waist': 90,
        'shoulder': 50
    }
}
def predict_size(height, weight, gender, chest=None, waist=None, shoulder=None):
    gender = gender.strip().lower()  
    if gender not in mean_values:
        raise ValueError("Giới tính phải là 'nam' hoặc 'nữ'.")
    # Điền giá trị thiếu bằng trung bình theo giới tính
    chest = chest if chest is not None else mean_values[gender]['chest']
    waist = waist if waist is not None else mean_values[gender]['waist']
    shoulder = shoulder if shoulder is not None else mean_values[gender]['shoulder']    
    # Tính BMI
    BMI = weight / ((height / 100) ** 2)
    # Tạo dataframe cho input
    input_df = pd.DataFrame([{
        'height': height,
        'weight': weight,
        'chest': chest,
        'waist': waist,
        'shoulder': shoulder,
        'gender': gender,
        'BMI': BMI
    }])

    # Dự đoán
    pred_encoded = pipeline.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    size_info = size_description[gender][pred_label]
    return pred_label, size_info['target_audience'], size_info['style']

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
            chest_input = input("Nhập vòng ngực (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            waist_input = input("Nhập vòng eo (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            shoulder_input = input("Nhập ngang vai (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            # Chuyển đổi nếu có
            chest = float(chest_input) if chest_input else None
            shoulder = float(shoulder_input) if shoulder_input else None
            waist = float(waist_input) if waist_input else None

            # Dự đoán
            result, target_audience, style = predict_size(height, weight, gender, chest, waist, shoulder)
            print(f"\n🎯 Recommended size: {result.upper()}")
            print(f"👥 Target audience: {target_audience}")
            print(f"👗 Style: {style}")
    except Exception as e:
        print("❌ Lỗi khi nhập hoặc xử lý dữ liệu:", e)
