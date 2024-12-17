import os
import sys
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

class MyLogger:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        # 确保日志目录存在
        log_dir = os.path.dirname(os.path.abspath(self.logfile_path))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _write_to_file(self, level, message):
        # 写入到文件
        with open(self.logfile_path, 'a', encoding='utf-8') as f:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{time_str}] [{level}] {message}\n")

    def _print_to_console(self, color, level, message):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 在控制台打印带颜色的日志
        print(f"{color}[{time_str}] [{level}] {message}{Style.RESET_ALL}")

    def log(self, message):
        self._write_to_file("INFO", message)
        self._print_to_console(Fore.WHITE, "INFO", message)

    def warn(self, message):
        self._write_to_file("WARN", message)
        self._print_to_console(Fore.YELLOW, "WARN", message)

    def error(self, message):
        self._write_to_file("ERROR", message)
        self._print_to_console(Fore.RED, "ERROR", message)


if __name__ == "__main__":
    # 示例使用
    logger = MyLogger("logs/app.log")
    logger.log("这是普通日志信息。")
    logger.warn("这是警告日志信息。")
    logger.error("这是错误日志信息。")
