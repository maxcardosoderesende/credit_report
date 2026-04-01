from loguru import logger
import sys


logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/pipeline.log", rotation="10 MB")