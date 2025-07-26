from datetime import datetime
import pytz
from loguru import logger


def get_date_time():
    return datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))
