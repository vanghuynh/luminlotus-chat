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

size_description = {
    'nữ': {
        'S': {"target_audience": "Thích mặc áo rộng rãi, thoải mái hơn size XS, hoặc dáng người vừa phải.", "style": "Trẻ trung, thoải mái, dễ phối đồ."},
        'M': {"target_audience": "Phù hợp với đa số nữ giới muốn sự thoải mái.", "style": "Rộng vừa phải, thoải mái tối đa, che khuyết điểm nhẹ."},
        'L': {"target_audience": "Thích mặc rộng rãi, thoải mái hoặc có vóc dáng đầy đặn hơn.", "style": "Tạo sự thoải mái tối đa, phóng khoáng, có thể mặc dáng oversized nhẹ."},
        'XL': {"target_audience": "Rất cao, hoặc muốn mặc áo thật rộng rãi, phom phóng khoáng.", "style": "Thoải mái tối đa, phom dáng rộng, phù hợp phong cách cá tính."},
        '2XL': {"target_audience": "Vóc dáng lớn hoặc rất cao, muốn mặc áo siêu rộng (oversized).", "style": "Thoải mái vượt trội, phong cách độc đáo, ấn tượng."}
    },
    'nam': {
        'S': {"target_audience": "Vóc dáng vừa phải, thích mặc áo ôm vừa vặn.", "style": "Lịch sự, gọn gàng, năng động."},
        'M': {"target_audience": "Phù hợp với đa số nam giới, muốn áo vừa vặn, thoải mái.", "style": "Vừa vặn, không quá ôm cũng không quá rộng, năng động."},
        'L': {"target_audience": "Vóc dáng trung bình khá đến đầy đặn, muốn áo rộng rãi.", "style": "Thoải mái, phóng khoáng, dễ vận động."},
        'XL': {"target_audience": "Vóc dáng cao to.", "style": "Rộng rãi, thoải mái tối đa, phù hợp phong cách Streetwear."},
        '2XL': {"target_audience": "Vóc dáng lớn, cao trên 1m80 và/hoặc cân nặng trên 100kg.", "style": "Thoải mái vượt trội, phong cách thể thao hoặc cá tính mạnh mẽ."}
    }
}

# Hàm dự đoán size với giá trị mặc định cho các trường phụ
def predict_size(height, weight, gender, age=None, length_back=None, chest=None, ngang_vai=None, vong_eo=None):
    bmi = weight / ((height / 100) ** 2)

    # Gán giá trị mặc định nếu không có
    if age is None:
        age = 25
    if length_back is None:
        length_back = 72.0
    if chest is None:
        chest = 100.0
    if ngang_vai is None:
        ngang_vai = 40.0  # Set default value for ngang_vai (should be adjusted based on domain knowledge)
    if vong_eo is None:
        vong_eo = 80.0  # Set default value for vong_eo (should be adjusted based on domain knowledge)

    input_df = pd.DataFrame([[height, weight, age, gender.lower(), bmi, length_back, chest, ngang_vai, vong_eo]],
                             columns=['height', 'weight', 'age', 'gender', 'BMI', 'length_back', 'chest', 'ngang_vai', 'vong_eo'])

    # Make the prediction
    pred_encoded = model.predict(input_df)[0]
    predicted_size = le.inverse_transform([int(pred_encoded)])[0]
    # Get the style and target audience based on the predicted size and gender
    size_info = size_description[gender][predicted_size]
    return predicted_size, size_info['target_audience'], size_info['style']
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
            ngang_vai_input = input("Nhập ngang vai (cm, tùy chọn, Enter nếu bỏ qua): ").strip()
            vong_eo_input = input("Nhập vòng eo (cm, tùy chọn, Enter nếu bỏ qua): ").strip()

            # Chuyển đổi nếu có
            age = int(age_input) if age_input else None
            length_back = float(length_back_input) if length_back_input else None
            chest = float(chest_input) if chest_input else None
            ngang_vai = float(ngang_vai_input) if ngang_vai_input else None
            vong_eo = float(vong_eo_input) if vong_eo_input else None

            # Dự đoán
            size, target_audience, style = predict_size(height, weight, gender, age, length_back, chest, ngang_vai, vong_eo)
            print(f"\n🎯 Recommended size: {size}")
            print(f"👥 Target audience: {target_audience}")
            print(f"👗 Style: {style}")
    except Exception as e:
        print("❌ Lỗi khi nhập hoặc xử lý dữ liệu:", e)
