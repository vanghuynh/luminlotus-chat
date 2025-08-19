from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from src.agents.rag_agent_template.func import agent
import os

# Load biến môi trường
load_dotenv()

app = Flask(__name__)
CORS()

# Route hiển thị giao diện HTML
@app.route("/")
def index():
    return render_template("ui.html")

# Route xử lý chat
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        country_code = (data.get("country_code") or "").strip()

        if not user_message:
            return jsonify({"error": "Missing message"}), 400

        # Nếu là yêu cầu greeting từ frontend
        if user_message == "init_greeting":
            intro = (
                "👋 <b>Chào bạn!</b> Tôi là <b>AIFSHOP</b> – trợ lý mua sắm thời trang của bạn.<br>"
                "👉 Bạn cần tôi giúp gì hôm nay? (gợi ý size, tìm sản phẩm, kiểm tra đơn, mã giảm giá...)<br>"
                "<b>👋 Hello!</b> I am <b>AIFSHOP</b> – your fashion shopping assistant.<br>"
                "👉 How can I assist you today? (size suggestions, product search, order tracking, discount codes...)"
            )
            return jsonify({"response": intro})

        # Gọi AI agent
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
        return jsonify({"response": response["messages"][-1].content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

application = app
