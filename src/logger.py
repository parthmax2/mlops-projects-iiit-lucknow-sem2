import os
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Logger configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='w',
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)
