# pylint: skip-file
# avoid parially initialized module error
from openai_manager.auth_manager import OpenAIAuthManager  # noqa
GLOBAL_MANAGER = OpenAIAuthManager()  # noqa


from openai_manager.api_resources import Completion, Embedding, ChatCompletion  # noqa
import logging
import os


logging.basicConfig(level=int(os.getenv("OPENAI_LOG_LEVEL", logging.WARNING)))
logger = logging.getLogger(__name__)
logger.debug(f"Logger level: {logger.level}")


def append_auth_from_config(config_path=None, config_dicts=None):
    GLOBAL_MANAGER.append_auth_from_config(config_path, config_dicts)


__all__ = [
    "Completion", 
    "Embedding", 
    "ChatCompletion"
]

name = "openai-manager"
