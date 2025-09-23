from pprint import pp
from unittest import result
from langchain_core.tools import tool
from src.utils.helper import convert_list_context_source_to_str
from src.utils.logger import logger
from langchain_core.runnables import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from src.utils.rcmsizetool import predict_size_public_text
import psycopg2
import re
from dotenv import load_dotenv
import os
import json
from flask import request
from urllib.parse import quote
from typing import Optional, Tuple

duckduckgo_search = DuckDuckGoSearchRun()
python_exec = PythonREPL()
load_dotenv()
conn_str = os.getenv("SUPABASE_DB_URL")
conn = psycopg2.connect(conn_str)
cursor = conn.cursor()
BASE_URL = os.getenv("PRODUCT_BASE_URL")

# Hàm gợi ý size
def predict_size_model(user_text: str) -> str:
    """
    Gợi ý size dựa vào chiều cao, cân nặng, giới tính, phong cách mặc (ôm, vừa, rộng)
    Nếu người dùng sử dụng tiếng anh để giao tiếp thì kết quả trả ra bằng tiếng anh, nếu người dùng sử dụng tiếng việt để giao tiếp thì kết quả trả ra bằng tiếng việt.
    Args:
        height : chiều cao của người dùng
        weight: cân nặng của người dùng
        gender : giới tính của người dùng
        fit : phong cách như ôm, vừa, rộng
    """
    return predict_size_public_text(user_text)


# Hàm chuẩn hóa đầu và tìm kiếm sản phẩm
# Chuẩn hóa input
def normalize_size(size: str) -> str:
    if not size:
        return ""
    return size.strip().upper()
    
CATEGORY_MAPPING_INPUT = {
    "men": "Thời trang nam",
    "male": "Thời trang nam",
    "man": "Thời trang nam",
    "women": "Thời trang nữ",
    "female": "Thời trang nữ",
    "woman": "Thời trang nữ",
}
CATEGORY_MAPPING_OUTPUT = {
    "Thời trang nam": "Men's Fashion",
    "Thời trang nữ": "Women's Fashion",
}

def normalize_category(category: Optional[str]) -> str:
    if not category:
        return ""
    c = category.strip().lower()
    if c in CATEGORY_MAPPING_INPUT:
        return CATEGORY_MAPPING_INPUT[c]
    if "nam" in c or "men" in c:
        return "Thời trang nam"
    if "nữ" in c or "nu" in c or "women" in c or "woman" in c:
        return "Thời trang nữ"
    return category.title()

def translate_category_for_output(category_name: str, lang: str) -> str:
    if lang == "en":
        return CATEGORY_MAPPING_OUTPUT.get(category_name, category_name)
    return category_name

def normalize_lang(lang: Optional[str]) -> str:
    return (lang or "").strip().lower()

def detect_lang(user_input: Optional[str]) -> str:
    """Nhận diện nhanh EN/VN từ text người dùng (không dùng lib ngoài)."""
    t = (user_input or "").strip().lower()
    if any(x in t for x in ["en", "eng", "english", "us", "america", "american"]):
        return "en"
    if any(x in t for x in ["vi", "vn", "vie", "vietnam", "vietnamese", "tiếng việt", "tieng viet"]):
        return "vi"
    return "vi"  # mặc định

def format_price(amount: Optional[float], currency_symbol: Optional[str], country_code: str) -> str:
    if amount is None:
        return "Liên hệ"
    # Ưu tiên symbol từ DB; fallback theo country
    symbol = currency_symbol or ("$" if country_code == "US" else "₫")
    if country_code == "US":
        return f"{symbol}{amount:,.2f}"
    return f"{amount:,.0f} {symbol}"

# Parse khoảng giá tự nhiên
def parse_price_range(price_range: str) -> Optional[tuple[int, int]]:
    """
    Hỗ trợ cả tiếng Việt & tiếng Anh: 'dưới 500k', 'trên 1tr', 'từ 200k-500k',
    'under 50', 'over 100', 'about 70', 'from 20-40', '20-40', ...
    Trả về (min, max) hoặc None nếu không parse được.
    """
    if not price_range:
        return None
    pr_l = price_range.lower()
    # chuẩn hóa số: bỏ . , đổi 'tr' thành 000000, 'k' thành 000
    t = pr_l.replace(".", "").replace(",", "")
    t = t.replace("tr", "000000").replace("k", "000")
    t = re.sub(r"[^\d\-]", " ", t)
    digits = [int(s) for s in t.split() if s.isdigit()]

    price_min, price_max = 0, 1_000_000_000  # 1e9
    if ("dưới" in pr_l or "under" in pr_l) and digits:
        price_max = digits[0]
    elif ("trên" in pr_l or "over" in pr_l) and digits:
        price_min = digits[0]
    elif ("khoảng" in pr_l or "about" in pr_l) and digits:
        price_min = price_max = digits[0]
    elif ("từ" in pr_l or "from" in pr_l) and "-" in pr_l:
        try:
            a, b = pr_l.split("-", 1)
            price_min = int("".join(filter(str.isdigit, a)))
            price_max = int("".join(filter(str.isdigit, b)))
        except Exception:
            pass
    elif "-" in pr_l:
        try:
            a, b = pr_l.split("-", 1)
            price_min = int("".join(filter(str.isdigit, a)))
            price_max = int("".join(filter(str.isdigit, b)))
        except Exception:
            pass
    else:
        # Không nhận ra pattern → trả None để bỏ filter
        return None
    return (price_min, price_max)

def unknown_label(lang: str) -> str:
    return "Unknown" if (lang or "").lower() == "en" else "Chưa rõ"

# ==== Hàm tìm kiếm sản phẩm ====
def extract_query_product(
    size: str = "",
    color: str = "",
    price_range: str = "",
    in_stock: bool = True,
    limit: int = 5,
    country_code: str = "",
    lang: str = "",
    category_name: str = "",
) -> str:
    """
    Truy vấn sản phẩm 
    Tự động nhận diện lang từ input (en/vi), map sang country_code (US/VN) nếu chưa truyền.
    Hiển thị đúng ký hiệu tiền theo bảng Country hoặc fallback theo country_code.
    Điều kiện nào không muốn cung cấp thì bỏ qua (người dùng có thể nói "nào cũng được"/"any"/"no preference/"không"/"No"/"no", tất cả sản phẩm" cho điều kiện đó).
    Returns:
        str: Kết quả tìm kiếm sản phẩm dưới dạng markdown.
    """
    lang = normalize_lang(detect_lang(lang))  

    # Chuẩn hóa input
    size = normalize_size(size)
    category_name_norm = normalize_category(category_name)

    if not country_code:
        country_code = "US" if lang == "en" else "VN"

    sql = """
    SELECT 
        p.id,
        p.name AS product_name,
        pp.price,
        c."currencySymbol",
        v.size,
        v.color,
        v.sku,
        v.stock,
        p.images[1] AS image_url,
        cat.name AS category_db_name
    FROM "Product" p
    LEFT JOIN "ProductVariant" v ON v."productId" = p.id
    LEFT JOIN "ProductPrice" pp ON pp."productId" = p.id
    LEFT JOIN "Country" c ON c.id = pp."countryId"
    LEFT JOIN "Category" cat ON cat.id = p."categoryId"
    WHERE 1=1
    """
    params = []

    # Lọc quốc gia
    sql += " AND c.code = %s"
    params.append(country_code)

    # Lọc còn hàng
    if in_stock:
        sql += " AND v.stock > 0"

    has_filters = False

    # Lọc danh mục
    if category_name_norm:
        has_filters = True
        sql += " AND LOWER(cat.name) LIKE %s"
        params.append(f"%{category_name_norm.lower()}%")

    # Lọc size
    if size:
        has_filters = True
        sql += " AND UPPER(v.size) = %s"
        params.append(size)

    # Lọc màu
    if color:
        has_filters = True
        sql += " AND LOWER(v.color) = %s"
        params.append(color.lower())

    # Giá
    price = parse_price_range(price_range)
    if price:
        has_filters = True
        price_min, price_max = price
        sql += " AND pp.price BETWEEN %s AND %s"
        params.extend([price_min, price_max])

    # Sắp xếp
    if not has_filters:
        sql += ' ORDER BY p."createdAt" DESC, p.id DESC LIMIT %s'
    else:
        sql += ' ORDER BY COALESCE(pp.price, p.price) ASC, p."createdAt" DESC LIMIT %s'
    params.append(limit)

    # Thực thi query
    cursor.execute(sql, params)
    products = cursor.fetchall()

    if not products:
        return "😔 Không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn."

    response = "🔎 **Kết quả tìm kiếm sản phẩm:**\n"
    for p in products:
        pid, name, price, currency_symbol, size, color, sku, stock, images_url, category_db_name = p
        # Dịch danh mục nếu lang = 'en'
        cat_display = translate_category_for_output(
            normalize_category(category_db_name or ""), lang
        ) or unknown_label(lang)        
        price_fmt = format_price(price, currency_symbol, country_code)
        response += (
            f"\n🧥 **{name}**\n"
            f"- Danh mục: {cat_display}\n"
            f"- 💰 Giá: {price_fmt}\n"
            f"- 🎨 Màu: {color} | 📏 Size: {size}\n"
            f"- 🔢 SKU: {sku} | 📦 Có sẵn: {stock}\n"
            f"- [Xem chi tiết]({BASE_URL}/{pid})\n"
            f"- 🖼️ Hình ảnh: ![Image]({images_url})\n"
        )
    response += "\n👉 Bạn muốn xem chi tiết sản phẩm nào không?"
    return response

# Trích xuất kiểm tra đơn hàng
def check_order_status(
    order_id: str = "", country_code: str = "", lang: str = ""
) -> str:
    """
    Kiểm tra tình trạng đơn hàng và hiển thị chi tiết từng đơn hàng kèm thông tin khách hàng và sản phẩm theo mã đơn hàng.
    Args:
        order_id (str): Mã đơn hàng của đơn hàng.
    Returns:
        str: Kết quả kiểm tra đơn hàng dưới dạng markdown.
    """
    if not country_code:
        country_code = "US" if lang == "en" else "VN"
    price_unit = "$" if lang == "en" else "VND"
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
            a."countryId",
            o."uniqueCode"
        FROM "Order" o
        LEFT JOIN "Address" a ON a.id = o."addressId"
        WHERE 1=1
    """
    params = []
    if order_id:
        sql += ' AND o."uniqueCode" ILIKE %s'
        params.append(f"%{order_id}%")
    try:
        cursor.execute(sql, params)
        orders = cursor.fetchall()
        if not orders:
            return "Không tìm thấy đơn hàng nào khớp với thông tin bạn cung cấp."
        response = "Tôi đã tìm thấy các đơn hàng của bạn với thông tin đã cung cấp:\n"
        for order in orders:
            (
                order_db_id,
                order_code,
                status,
                created_at,
                total,
                shipping_full_name,
                first_name,
                last_name,
                email,
                phone_num,
                note,
                street,
                ward,
                district,
                province,
                country_id,
                unique_code,
            ) = order
            full_name = shipping_full_name.strip() if shipping_full_name else f"{first_name or ''} {last_name or ''}".strip()
            created_at_fmt = created_at.strftime("%d/%m/%Y")
            total_fmt = f"{total:,.0f} {price_unit}"
            note = note if note else "(không có ghi chú)"
            shipping_address = ", ".join([p for p in [street, ward, district, province] if p])
            # Lấy sản phẩm (dùng cách lấy hình như extract_query_product)
            item_sql = """
                SELECT 
                    p.name,
                    v.size,
                    v.color,
                    i.quantity,
                    i.price,
                    COALESCE(p.images[1], '') AS image_url
                FROM "OrderItem" i
                JOIN "Product" p ON p.id = i."productId"
                JOIN "ProductVariant" v ON v.id = i."productVariantId"
                WHERE i."orderId" = %s
            """
            cursor.execute(item_sql, (order_db_id,))
            items = cursor.fetchall()
            response += f"\n**Đơn hàng #{unique_code}**\n"
            # Hiển thị sản phẩm trước, mỗi thuộc tính xuống dòng
            for name, size, color, quantity, price, image_url in items:
                price_fmt = f"{price:,.0f} {price_unit}"
                response += (
                    f"* Sản phẩm: {name}\n"
                    f"  - Size: {size}\n"
                    f"  - Màu: {color}\n"
                    f"  - Số lượng: {quantity}\n"
                    f"  - Giá: {price_fmt}\n"
                )
                if image_url:
                    response += f"  - 🖼️ Hình ảnh: ![Image]({image_url})\n"
            # Sau đó mới tới thông tin đơn hàng
            response += (
                f"- Trạng thái: {status}\n"
                f"- Ngày đặt: {created_at_fmt}\n"
                f"- Tổng tiền: {total_fmt}\n"
                f"- Người nhận: {full_name}\n"
                f"- Email: {email}\n"
                f"- Số điện thoại: {phone_num}\n"
                f"- Địa chỉ giao hàng: {shipping_address}\n"
                f"- Ghi chú: {note}\n"
                f"- Mã đơn hàng duy nhất: {unique_code}\n"
            )
        return response
    except Exception as e:
        logger.error(f"Error checking order: {e}")
        return "Đã xảy ra lỗi khi kiểm tra đơn hàng. Vui lòng thử lại sau."

# Truy vấn thông tin chi tiết sản phẩm
def extract_information_product(
    product_keyword: str, lang: str = "vi", country_code: str = ""
) -> str:
    """
    Truy vấn thông tin chi tiết sản phẩm theo từ khóa hoặc tên sản phẩm.
    Args:
        product_keyword (str): Từ khóa hoặc tên sản phẩm cần tìm.
        lang (str): Ngôn ngữ của người dùng, ảnh hưởng đến cách hiển thị kết quả.
        country_code (str): Mã quốc gia để lấy giá theo quốc gia.
    Returns:
        str: Kết quả thông tin chi tiết sản phẩm dưới dạng markdown.
    """
    if not country_code:
        country_code = "US" if lang == "en" else "VN"
    price_unit = "$" if lang == "en" else "VND"
    sql = """
        SELECT 
            p.id,
            p.name,
            p.description,
            pp.price AS country_price,
            p.images[1] AS image_url,
            c.name AS category_name,
            v.id AS variant_id,
            v.color,
            v.size,
            v.stock AS variant_stock,
            v.sku,
            v.weight
        FROM "Product" p
        LEFT JOIN "ProductVariant" v ON v."productId" = p.id 
        LEFT JOIN "Category" c ON c.id = p."categoryId"
        LEFT JOIN "ProductPrice" pp ON pp."productId" = p.id
        LEFT JOIN "Country" co ON co.id = pp."countryId"
        WHERE LOWER(p.name) ILIKE %s AND co.code = %s
        ORDER BY v.size, v.color
    """
    cursor.execute(sql, (f"%{product_keyword.lower()}%", country_code))
    rows = cursor.fetchall()
    if not rows:
        return f"Không tìm thấy sản phẩm nào khớp với từ khóa: {product_keyword}"
    first = rows[0]
    name, desc, price, images_url, category = (
        first[1],
        first[2],
        first[3],
        first[4],
        first[5],
    )
    header_variant_stock = first[9]
    response = f"🛍 **{name}**\n"
    response += (
        f"- Danh mục: {category}\n"
        f"- Giá: {price:,.0f} {price_unit}\n"
        f"- Có sẵn: {header_variant_stock}\n"
        f"- Mô tả: {desc}\n"
        f"- [Xem chi tiết]({BASE_URL}/{first[0]})\n"
    )
    response += f"- 🖼️ Hình ảnh: ![Image]({images_url})\n"
    response += "\n🔄 **Các biến thể:**\n"
    for row in rows:
        response += (
            f"* Màu: {row[7]} \n"
            f"  Size: {row[8]} – \n"
            f"  SKU: {row[10]} – \n"
            f"  Có sẵn: {row[9]} – \n"
            f"  Nặng: {row[11]}kg\n"
        )
    return response


# Lấy thông tin về các chương trình khuyến mãi hiện có
def check_active_coupons(lang: str = "", country_code: str = "") -> str:
    """
    Trả về danh sách các mã giảm giá còn hiệu lực dưới dạng markdown.
    """
    if not country_code:
        country_code = "US" if lang == "en" else "VN"
    price_unit = "$" if lang == "en" else "VND"
   
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
            AND type = 'PUBLIC'
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
                discount_info = f"{discount_value:,.0f} {price_unit}"
            elif discount_type == "PERCENT":
                discount_info = f"{discount_value:.0f}%"
            else:
                discount_info = f"{discount_value}"

            max_discount_str = (
                f" – Giảm tối đa {max_discount:,.0f} {price_unit}"
                if max_discount
                else ""
            )
            min_order_str = (
                f" – Đơn tối thiểu {min_order_value:,.0f} {price_unit}"
                if min_order_value
                else ""
            )

            response += (
                f"- Mã **{code}**: Giảm {discount_info}{max_discount_str}{min_order_str}\n"
                f"  ⏳ Hiệu lực: {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}\n"
            )

        return response

    except Exception as e:
        logger.error(f"Lỗi khi lấy coupon: {e}")
        return "Đã xảy ra lỗi khi kiểm tra mã giảm giá. Vui lòng thử lại sau."
