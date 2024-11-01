import logging
import os
from datetime import datetime

# We create filename using date function like this "09_03_2024_11_30_45.log"
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Now we join our path using os.path.join
logs_path = os.path.join(os.getcwd(),'logs',LOG_FILE)
os.makedirs(logs_path,exist_ok=True)  # create our new directory

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE) # join path with new directory and Log_file

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)


