from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from src.agents.rag_agent_template.func import agent
import os

# Load biến môi trường
load_dotenv()

app = Flask(__name__)
CORS(app)

# Route hiển thị giao diện HTML
@app.route("/")
def index():
    return render_template("ui.html")  # File này nằm trong thư mục templates/

already_greeted = False

# Route xử lý chat
@app.route("/chat", methods=["POST"])
def chat():
    global already_greeted

    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Missing message"}), 400
        # 👋 Nếu là tín hiệu mở trang, gửi lời chào mặc định
        if user_message == "init_greeting":
            already_greeted = True
            intro = (
                "👋 <b>Chào bạn!</b> Tôi là <b>AIFSHOP</b> – trợ lý mua sắm thời trang của bạn.<br>"
            )
            return jsonify({"response": intro})
        # Nếu chưa gửi lời chào → tự động gửi trước
        if not already_greeted:
            intro = (
                "👋 <b>Chào bạn!</b> Tôi là <b>AIFSHOP</b> – trợ lý mua sắm thời trang của bạn.<br>"
                "👉 Bạn cần tôi giúp gì hôm nay? (gợi ý size, tìm sản phẩm, kiểm tra đơn, mã giảm giá...)"
            )
            already_greeted = True
            return jsonify({"response": intro})
        # Nếu đã chào rồi → gửi vào agent như bình thường
        config = {"configurable": {"thread_id": "1"}}
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config
        )
        return jsonify({"response": response["messages"][-1].content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

application = app