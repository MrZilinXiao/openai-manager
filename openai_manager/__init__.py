# pylint: skip-file
# avoid parially initialized module error
from openai_manager.auth_manager import OpenAIAuthManager  # noqa
GLOBAL_MANAGER = OpenAIAuthManager()  # noqa


from openai_manager.api_resources import Completion  # noqa
from functools import partialmethod, partial  # noqa
import logging
import os


logging.basicConfig(level=int(os.getenv("OPENAI_LOG_LEVEL", logging.WARNING)))
logger = logging.getLogger(__name__)
logger.debug(f"Logger level: {logger.level}")

__all__ = [
    "Completion"
]

name = "openai-manager"
