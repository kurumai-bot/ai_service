from datetime import datetime
import logging
import os
from os import path
import time


LOGGER_FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
logging.Formatter.converter = time.gmtime

LOGGER = logging.getLogger("kurumai_service")
LOGGER.propagate = False
LOGGER.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(LOGGER_FORMATTER)
stream_handler.setLevel(logging.INFO)
LOGGER.addHandler(stream_handler)

if not path.isdir("logs"):
    os.mkdir("logs")
file_handler = logging.FileHandler(
    path.join("logs", datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + ".log")
)
file_handler.setFormatter(LOGGER_FORMATTER)
file_handler.setLevel(logging.DEBUG)
LOGGER.addHandler(file_handler)


HOST = "LOCALHOST"
PORT = 8079


with open("secrets", "r", encoding="utf-8") as secrets_file:
    AUTH_KEY = secrets_file.readline().removesuffix("\n").encode()
    OPENAI_API_KEY = secrets_file.readline().removesuffix("\n")
