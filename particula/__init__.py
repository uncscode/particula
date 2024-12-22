""" a simple, fast, and powerful particle simulator.

    particula is a simple, fast, and powerful particle simulator,
    or at least two of the three, we hope. It is a simple particle
    system that is designed to be easy to use and easy to extend.
    The goal is to have a robust aerosol (gas phase + particle phase)
    simulation system that can be used to answer scientific questions
    that arise for experiments and research discussions.

    The main features of particula are:
    ...

    More details to follow.
"""

from particula.logger_setup import setup

__version__ = "0.1.1"

# setup the logger
logger = setup()
# log the version of particula upon loading
logger.info("particula version %s loaded.", __version__)
