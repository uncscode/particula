"""Setup for logging in the particula package.

Based on setup from:
https://youtu.be/9L77QExPmI0?si=pSKsVyVh2dE8QxFA
https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging
"""

import os
import logging
import logging.config

logger = logging.getLogger("particula")  # define the parent logger

# get path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# add the logging directory to the path
log_dir = os.path.join(current_dir, "logging")

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
            "filename": os.path.join(log_dir, "particula.log"),
            "maxBytes": 25_000_000,
            "backupCount": 3
        }
    },
    "loggers": {
        "particula": {
            "level": "DEBUG",
            "handlers": ["file", "stderr"],
            "propagate": False
        },
        "root": {
            "level": "ERROR",
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
    os.makedirs(log_dir, exist_ok=True)
    logging.config.dictConfig(config)
    return logger
