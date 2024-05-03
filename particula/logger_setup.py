"""Setup for logging in the particula package.

Based on setup from:
https://youtu.be/9L77QExPmI0?si=pSKsVyVh2dE8QxFA
https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging
"""

import os
import logging
import logging.config

logger = logging.getLogger("particula")  # define the parent logger


config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(levelname)s|%(module)s|L%(lineno)d]: %(message)s"
        },
        "detailed": {
            "format":
            "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z"
        }
    },
    "handlers": {
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "simple",
            "stream": "ext://sys.stderr"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logging/particula.log",
            "maxBytes": 1000000,
            "backupCount": 3
        }
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": [
                "stderr",
                "file"
            ]
        }
    }
}


def setup():
    """Setup for logging in the particula package."""
    # check for logging directory
    logging_dir = "logging"
    try:
        os.mkdir(logging_dir)
    except FileExistsError:
        pass
    logging.config.dictConfig(config)
    return logger
