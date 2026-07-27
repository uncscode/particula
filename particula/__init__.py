"""Top-level package for the particula particle simulator.

This package exposes the primary aerosol, execution, runnable, and namespace
modules used by the public API while keeping optional backends out of the
import path.
"""

# pylint: disable=unused-import
# flake8: noqa
# pyright: basic

from particula import (
    gas,
    particles,
    util,
    dynamics,
    activity,
    equilibria,
)
from particula.aerosol import Aerosol
from particula.aerosol_builder import AerosolBuilder
from particula.execution import (
    Backend,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionAdapter,
    ExecutionContext,
    ExecutionRequest,
    Process,
)
from particula.runnable import RunnableSequence

from particula.logger_setup import setup

__version__ = "0.2.12"

# setup the logger
logger = setup()
# log the version of particula upon loading
logger.info("particula version %s loaded.", __version__)
