import os
import logging
import numpy as np
from dataclasses import dataclass


""" configurations blueprint"""

@dataclass(frozen=True)  # Instances of this class are immutable.
class Config:
    e = 0



def set_logger_config():
    """     Configures the logger for the application. """
    # Create a logger
    logger = logging.getLogger('OutageScheduler')
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('outage_scheduler.log')

    # Set level of handlers
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.ERROR)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger