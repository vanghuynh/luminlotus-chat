# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from dotenv import load_dotenv
# from src.agents.rag_agent_template.func import agent
# import os

# # Load biến môi trường
# load_dotenv()

# app = Flask(__name__)
# CORS(app)
    
# already_greeted = False

# # Route hiển thị giao diện HTML
# @app.route("/")
# def index():
#     return render_template("ui.html")  # File này nằm trong thư mục templates/

# # Route xử lý chat
# @app.route("/chat", methods=["POST"])
# def chat():
#     global already_greeted

#     try:
#         data = request.get_json()
#         user_message = data.get("message", "").strip()
#         country_code = data.get("country_code", "")  
#         if not user_message:
#             return jsonify({"error": "Missing message"}), 400
#         # 👋 Nếu là tín hiệu mở trang, gửi lời chào mặc định cả 2 ngôn ngữ
#         intro = (
#             "👋 Chào bạn! Tôi là AIFSHOP – trợ lý mua sắm thời trang của bạn.\n"
#             "👉 Bạn cần tôi giúp gì hôm nay? (gợi ý size, tìm sản phẩm, kiểm tra đơn, mã giảm giá...)\n"
#             "👋 Hello! I am AIFSHOP – your fashion shopping assistant.\n"
#             "👉 How can I assist you today? (size suggestions, product search, order tracking, discount codes...)\n"
#         )
        
#         # Nếu chưa gửi lời chào → tự động gửi trước
#         if not already_greeted:
#             intro = (
#                 "👋 <b>Chào bạn!</b> Tôi là <b>AIFSHOP</b> – trợ lý mua sắm thời trang của bạn.<br>"
#                 "👉 Bạn cần tôi giúp gì hôm nay? (gợi ý size, tìm sản phẩm, kiểm tra đơn, mã giảm giá....<br>)"
#                 "<b>👋 Hello!</b> I am <b>AIFSHOP</b> – your fashion shopping assistant.<br>"
#                 "👉 How can I assist you today? (size suggestions, product search, order tracking, discount codes...)"
#             )
#             already_greeted = True
#             return jsonify({"response": intro})
#         # Nếu đã chào rồi → gửi vào agent như bình thường
#         config = {
#             "configurable": {
#                 "thread_id": "1",
#                 "country_code": country_code
#             }
#         }
#         response = agent.invoke(
#             {"messages": [{"role": "user", "content": user_message}]},
#             config=config
#         )
#         return jsonify({"response": response["messages"][-1].content})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

# application = app

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
from src.agents.rag_agent_template.func import agent
import os

# Load biến môi trường
load_dotenv()

app = Flask(__name__)
# Khóa bí mật để dùng session cookie
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# Nếu frontend khác origin, cần supports_credentials=True
CORS(app, supports_credentials=True)

# Route hiển thị giao diện HTML
@app.route("/")
def index():
    # Mỗi lần người dùng mở trang mới, cho phép chào lại trong phiên này
    session["greeted"] = False
    return render_template("ui.html")  # File này nằm trong thư mục templates/

# Route xử lý chat
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        country_code = (data.get("country_code") or "").strip()

        if not user_message:
            return jsonify({"error": "Missing message"}), 400

        # Nội dung lời chào (HTML, để frontend render bằng innerHTML / |safe)
        intro = (
            "👋 <b>Chào bạn!</b> Tôi là <b>AIFSHOP</b> – trợ lý mua sắm thời trang của bạn.<br>"
            "👉 Bạn cần tôi giúp gì hôm nay? (gợi ý size, tìm sản phẩm, kiểm tra đơn, mã giảm giá....<br>)"
            "<b>👋 Hello!</b> I am <b>AIFSHOP</b> – your fashion shopping assistant.<br>"
            "👉 How can I assist you today? (size suggestions, product search, order tracking, discount codes...)"
        )

        # Nếu phiên hiện tại chưa chào -> chào trước, sau đó đánh dấu đã chào
        if not session.get("greeted", False):
            session["greeted"] = True
            return jsonify({"response": intro, "is_greeting": True})

        # Nếu đã chào rồi → gọi agent bình thường
        config = {
            "configurable": {
                "thread_id": "1",
                "country_code": country_code
            }
        }
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config
        )
        return jsonify({"response": response["messages"][-1].content, "is_greeting": False})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Với dev server, không nên bật reloader 2 process khi dùng session đơn giản
    app.run(host="0.0.0.0", port=5000, debug=True)

application = app
