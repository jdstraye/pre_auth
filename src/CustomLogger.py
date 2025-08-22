""" 
Creates a custom log handler that creates a fresh log file with a date in the name but also creates a symlink to the latest log file.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import os
# --- Custom Log Handler & Logging Setup (from previous code) ---
class CustomRotatingFileHandler(RotatingFileHandler):
    """
    A custom rotating file handler that appends a timestamp to the log file name.
    """
    def __init__(self, filename, maxBytes, backupCount):
        self.base_filename = Path(filename).stem
        self.base_dir = Path(filename).absolute().parent
        self.current_datetime = datetime.now().strftime("%Y%m%d.%H%M")
        self.baseFilename = str(self.base_dir / f"{self.base_filename}-{self.current_datetime}.log")
        super().__init__(self.baseFilename, maxBytes=maxBytes, backupCount=backupCount)
        self.latest_log_path = self.base_dir / f"{self.base_filename}_latest.log"
        if self.latest_log_path.exists() or self.latest_log_path.is_symlink():
            self.latest_log_path.unlink()
        self.latest_log_path.symlink_to(Path(self.baseFilename).relative_to(self.base_dir))

    def doRollover(self):
        """
        Modified rollover to handle the timestamped log file names correctly.
        """
        if self.stream:
            self.stream.close()
        current_datetime = datetime.now().strftime("%Y%m%d.%H%M")
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.base_dir}/{self.base_filename}-{current_datetime}.{i}.log"
                dfn = f"{self.base_dir}/{self.base_filename}-{current_datetime}.{i + 1}.log"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = f"{self.base_dir}/{self.base_filename}-{current_datetime}.1.log"
            if os.path.exists(dfn):
                os.remove(dfn)
            os.rename(self.baseFilename, dfn)
        self.baseFilename = f"{self.base_dir}/{self.base_filename}-{current_datetime}.log"
        self.stream = open(self.baseFilename, 'w')

