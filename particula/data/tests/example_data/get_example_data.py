"""get location of data folder"""

import os


def get_data_folder():
    """Get the location of the data folder."""
    path = os.path.dirname(os.path.abspath(__file__))
    return path
