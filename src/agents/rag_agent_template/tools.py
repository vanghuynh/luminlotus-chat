from unittest import result
from langchain_core.tools import tool
from src.utils.helper import convert_list_context_source_to_str
from src.utils.logger import logger
from langchain_core.runnables import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from src.utils.rcmsizetool import predict_size
import psycopg2
import re
from dotenv import load_dotenv
import os

duckduckgo_search = DuckDuckGoSearchRun()
python_exec = PythonREPL()
load_dotenv() 
conn_str = os.getenv("SUPABASE_DB_URL")
conn = psycopg2.connect(conn_str)
cursor = conn.cursor()

# Hàm dự đoán size quần áo
def predict_size_model(message: str) -> str:
    """
    Gợi ý size dựa trên các yếu tố như chiều cao, cân nặng, giới tính, tuổi, chiều dài lưng, vòng ngực.
    """
    try:
        height_match = re.search(r"cao\s*(\d+)", message)
        weight_match = re.search(r"nặng\s*(\d+)", message)
        gender_match = re.search(r"(nam|nữ|male|female)", message.lower())
        if not (height_match and weight_match and gender_match):
            return "Vui lòng cung cấp đầy đủ chiều cao, cân nặng và giới tính để tôi gợi ý size nhé, nếu có thêm chiều dài lưng và vòng ngực sẽ tốt hơn."
        height = float(height_match.group(1))
        weight = float(weight_match.group(1))
        gender = gender_match.group(1)
        # Thông tin tùy chọn
        length_back_match = re.search(r"(lưng|chiều dài lưng)\s*(\d+)", message)
        chest_match = re.search(r"(ngực|vòng ngực|ngang ngực)\s*(\d+)", message)
        length_back = float(length_back_match.group(2)) if length_back_match else None
        chest = float(chest_match.group(2)) if chest_match else None
        # Gọi model dự đoán
        result = predict_size(height, weight, gender, length_back, chest)
        response = (
                    f"📏 **Kết quả gợi ý size:**\n"
                    f"- Chiều cao: **{height}cm**\n"
                    f"- Cân nặng: **{weight}kg**\n"
                    f"- Giới tính: **{gender.capitalize()}**\n"
        )
        if length_back:
            response += f"- Chiều dài lưng: **{length_back}cm**\n"
        if chest:
            response += f"- Độ rộng ngực: **{chest}cm**\n"

        response += f"🎯 **Size phù hợp:** {result['recommended_size']}"
        return response

    except Exception as e:
        return f"Đã xảy ra lỗi khi xử lý: {e}"

# Tìm kiếm sản phẩm
def extract_query_product(
    size: str = "",
    color: str = "",
    price_range: str = "",
    in_stock: bool = True,
    limit: int = 5,
    country_code: str = "VN" 
) -> list:
    """
    Truy vấn sản phẩm theo kích cỡ, màu sắc, khoảng giá, còn hàng và giá theo quốc gia.
    """
    sql = """
    SELECT 
        p.id,
        p.name AS product_name,
        pp.price,
        v.size,
        v.color,
        v.sku,
        v.stock
    FROM "Product" p
    LEFT JOIN "ProductVariant" v ON v."productId" = p.id
    LEFT JOIN "ProductPrice" pp ON pp."productId" = p.id
    LEFT JOIN "Country" c ON c.id = pp."countryId"
    WHERE 1=1
    """
    params = []
    # ✅ Lọc quốc gia 
    sql += " AND c.code = %s"
    params.append(country_code)
    # Lọc theo size
    if size:
        sql += " AND v.size ILIKE %s"
        params.append(f"%{size.strip()}%")
    # Lọc theo màu sắc
    if color:
        sql += " AND v.color ILIKE %s"
        params.append(f"%{color.strip()}%")
    # Lọc theo còn hàng
    if in_stock:
        sql += " AND v.stock > 0"
    # Lọc theo khoảng giá
    price_min = 0
    price_max = 1e9
    if price_range:
        t = price_range.lower().replace(".", "").replace(",", "")
        digits = [int(s) for s in t.split() if s.isdigit()]
        if "dưới" in t and digits:
            price_max = digits[0]
        elif "trên" in t and digits:
            price_min = digits[0]
        elif "-" in t:
            try:
                parts = t.split("-")
                price_min = int("".join(filter(str.isdigit, parts[0])))
                price_max = int("".join(filter(str.isdigit, parts[1])))
            except:
                pass
    sql += " AND pp.price BETWEEN %s AND %s"
    params.extend([price_min, price_max])
    sql += " ORDER BY pp.price ASC LIMIT %s"
    params.append(limit)
    cursor.execute(sql, params)
    products = cursor.fetchall()

    if not products:
        return "😔 Không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn."

    response = "🔎 **Kết quả tìm kiếm sản phẩm:**\n"
    for p in products:
        pid, name, price, size, color, sku, stock = p
        response += (
            f"\n🧥 **{name}**\n"
            f"- 💰 Giá: {price:,.0f} VND\n"
            f"- 🎨 Màu: {color} | 📏 Size: {size}\n"
            f"- 🔢 SKU: {sku} | 📦 Tồn kho: {stock}\n"
        )
    response += "\n👉 Bạn muốn xem chi tiết sản phẩm nào không?"
    return response

# Trích xuất kiểm tra đơn hàng
def check_order_status(order_id: str = "", phone: str = "") -> str:
    """
    Kiểm tra tình trạng đơn hàng và hiển thị chi tiết từng đơn hàng kèm thông tin khách hàng và sản phẩm.
    """
    sql = """
        SELECT 
            o.id,
            o."orderCode",
            o.status,
            o."createdAt",
            o.total,
            o."shippingFullName",
            a."firstName",
            a."lastName",
            o."shippingEmail",
            a.phone,
            o."customerNote",
            a.street,
            a.ward,
            a.district,
            a.province,
            a."countryId"
        FROM "Order" o
        LEFT JOIN "Address" a ON a.id = o."addressId"
        WHERE 1=1
    """
    params = []
    if order_id:
        sql += " AND o.\"orderCode\" ILIKE %s"
        params.append(f"%{order_id}%")
    if phone:
        sql += " AND a.phone ILIKE %s"
        params.append(f"%{phone}%")
    sql += " ORDER BY o.\"createdAt\" DESC LIMIT 3"
    try:
        logger.info(f"Checking order status with params: {params}")
        cursor.execute(sql, params)
        orders = cursor.fetchall()
        if not orders:
            return "Không tìm thấy đơn hàng nào khớp với thông tin bạn cung cấp."
        response = "Tôi đã tìm thấy các đơn hàng của bạn với thông tin đã cung cấp:\n"
        
        for order in orders:
            (
                order_id,
                order_code,
                status,
                created_at,
                total,
                shipping_full_name,
                first_name,
                last_name,
                email,
                phone,
                note,
                street,
                ward,
                district,
                province,
                country_id
            ) = order
            # Xử lý tên người nhận
            if shipping_full_name:
                full_name = shipping_full_name.strip()
            else:
                full_name = f"{first_name or ''} {last_name or ''}".strip()
            created_at_fmt = created_at.strftime("%d/%m/%Y %H:%M")
            total_fmt = f"{total:,.0f} VND"
            note = note if note else "(không có ghi chú)"
            address_parts = [street, ward, district, province]
            shipping_address = ', '.join([p for p in address_parts if p])
            response += (
                f"\n**Đơn hàng {order_code}:**\n"
                f"* **Trạng thái:** {status}\n"
                f"* **Ngày đặt:** {created_at_fmt}\n"
                f"* **Tổng tiền:** {total_fmt}\n"
                f"* **Tên người đặt:** {full_name}\n"
                f"* **Số điện thoại:** {phone}\n"
                f"* **Email:** {email or '(không có email)'}\n"
                f"* **Địa chỉ:** {shipping_address}\n"
                f"* **Ghi chú:** {note}\n"
                f"* **Sản phẩm:**\n"
            )
            # Truy vấn sản phẩm của đơn hàng
            item_sql = """
                SELECT 
                    p.name,
                    v.size,
                    v.color,
                    i.quantity,
                    i.price
                FROM "OrderItem" i
                JOIN "Product" p ON p.id = i."productId"
                JOIN "ProductVariant" v ON v.id = i."productVariantId"
                WHERE i."orderId" = %s
            """
            cursor.execute(item_sql, (order_id,))
            items = cursor.fetchall()
            for item in items:
                name, size, color, quantity, price = item
                response += (
                    f"*   {name} (Size {size}, Màu {color}) – "
                    f"Số lượng: {quantity} – Giá: {price:,.0f} VND\n"
                )
        return response
    except Exception as e:
        logger.error(f"Error checking order: {e}")
        return "Đã xảy ra lỗi khi kiểm tra đơn hàng. Vui lòng thử lại sau."

# Lấy thông tin chi tiết về sản phẩm
def extract_information_product(product_keyword: str) -> str:
    """
    Trả về thông tin chi tiết sản phẩm bao gồm các biến thể: size, màu, SKU, tồn kho, giá, khối lượng.
    """
    sql = """
        SELECT 
        p.id,
        p.name,
        p.description,
        p.price AS default_price,
        p.stock,
        p.images,
        c.name AS category_name,
        v.id AS variant_id,
        v.color,
        v.size,
        v.stock AS variant_stock,
        v.sku,
        v.weight
    FROM "Product" p
    LEFT JOIN "Category" c ON c.id = p."categoryId"
    LEFT JOIN "ProductVariant" v ON v."productId" = p.id 
    WHERE LOWER(p.name) ILIKE %s
    ORDER BY v.size, v.color
    """
    try:
        cursor.execute(sql, (f"%{product_keyword.lower()}%",))
        rows = cursor.fetchall()

        if not rows:
            return f"Không tìm thấy sản phẩm nào khớp với từ khóa: {product_keyword}"

        first_row = rows[0]
        product_name = first_row[1]
        description = first_row[2]
        default_price = first_row[3]
        total_stock = first_row[4]
        images = first_row[5]
        category_name = first_row[6]

        response = f"🛍 **{product_name}**\n"
        response += f"- Danh mục: {category_name}\n"
        response += f"- Giá mặc định: {default_price:,.0f} VND\n"
        response += f"- Tổng tồn kho: {total_stock} sản phẩm\n"
        response += f"- Mô tả: {description}\n"
        if images:
            response += f"- Hình ảnh: {images[0]}\n"

        response += "\n**🔄 Biến thể sản phẩm:**\n"
        for row in rows:
            color = row[8]
            size = row[9]
            variant_stock = row[10]
            sku = row[11]
            weight = row[12]

            response += (
                f"* Màu: {color} – Size: {size} – SKU: {sku} – "
                f"Tồn kho: {variant_stock} – Nặng: {weight}kg\n"
            )

        return response

    except Exception as e:
        logger.error(f"Lỗi khi truy vấn chi tiết sản phẩm: {e}")
        return "Đã xảy ra lỗi khi lấy chi tiết sản phẩm. Vui lòng thử lại."


# Lấy thông tin về các chương trình khuyến mãi hiện có
def check_active_coupons() -> str:
    """
    Trả về danh sách các mã giảm giá còn hiệu lực.
    """
    sql = """
        SELECT 
            code,
            type,
            "discountType",
            "discountValue",
            "maxDiscount",
            "minOrderValue",
            "startDate",
            "endDate"
        FROM "Coupon"
        WHERE "isActive" = TRUE
          AND NOW() BETWEEN "startDate" AND "endDate"
        ORDER BY "startDate" DESC
        LIMIT 10
    """
    try:
        cursor.execute(sql)
        coupons = cursor.fetchall()

        if not coupons:
            return "Hiện tại không có mã giảm giá nào đang hoạt động."

        response = "🎁 **Danh sách mã giảm giá hiện có:**\n"
        for c in coupons:
            (
                code,
                ctype,
                discount_type,
                discount_value,
                max_discount,
                min_order_value,
                start_date,
                end_date,
            ) = c

            # Hiển thị mức giảm
            if discount_type == "AMOUNT":
                discount_info = f"{discount_value:,.0f} VND"
            elif discount_type == "PERCENT":
                discount_info = f"{discount_value:.0f}%"
            else:
                discount_info = f"{discount_value}"

            max_discount_str = (
                f" – Giảm tối đa {max_discount:,.0f} VND" if max_discount else ""
            )
            min_order_str = (
                f" – Đơn tối thiểu {min_order_value:,.0f} VND" if min_order_value else ""
            )

            response += (
                f"- Mã **{code}**: Giảm {discount_info}{max_discount_str}{min_order_str}\n"
                f"  ⏳ Hiệu lực: {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}\n"
            )

        return response

    except Exception as e:
        logger.error(f"Lỗi khi lấy coupon: {e}")
        return "Đã xảy ra lỗi khi kiểm tra mã giảm giá. Vui lòng thử lại sau."
