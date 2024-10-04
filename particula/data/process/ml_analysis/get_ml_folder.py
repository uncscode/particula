"""get machine learning location of data folder"""

import os


def get_ml_analysis_folder():
    """Get the location of the data folder."""
    _path = os.path.dirname(os.path.abspath(__file__))
    return _path
