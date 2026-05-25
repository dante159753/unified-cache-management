#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from collections.abc import Hashable


class Logger:
    def __init__(self, name: str = "UC"):
        self.name = name

    def isEnabledFor(self, levelno: int) -> bool:
        return False

    def log(self, levelno, message, *args, exc_info=None, scope=None, rate_limit=False):
        return None

    def debug(self, message: str, *args, **kwargs):
        return None

    def info(self, message: str, *args, **kwargs):
        return None

    def warning(self, message: str, *args, **kwargs):
        return None

    def error(self, message: str, *args, **kwargs):
        return None

    def critical(self, message: str, *args, **kwargs):
        return None

    def exception(self, message: str, *args: Hashable, **kwargs: Hashable):
        return None

    def info_once(self, message: str, *args: Hashable, **kwargs: Hashable):
        return None

    def warning_once(self, message: str, *args: Hashable, **kwargs: Hashable):
        return None

    def debug_once(self, message: str, *args: Hashable, **kwargs: Hashable):
        return None

    def info_limit(self, message: str, *args, **kwargs):
        return None

    def warning_limit(self, message: str, *args, **kwargs):
        return None

    def debug_limit(self, message: str, *args, **kwargs):
        return None


def init_logger(name: str = "UC") -> Logger:
    return Logger(name)


def current_formatter_type(lgr):
    return None


if __name__ == "__main__":
    logger = init_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
