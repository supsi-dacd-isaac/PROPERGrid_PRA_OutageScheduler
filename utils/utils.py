from pathlib import Path
import logging


# ANSI escape code for colored text
blue_c, green_c, purple_c, cyan_c, red_c, gray_c = "\033[94m", "\033[92m", "\033[95m", "\033[96m", "\033[91m", " "
bold_c, underline_c, reset_c = "\033[1m", "\033[4m", "\033[0m"

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO)


def get_project_root() -> Path:
    """get project root path"""
    return Path(__file__).parent.parent


