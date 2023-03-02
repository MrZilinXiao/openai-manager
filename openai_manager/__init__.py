from openai_manager.auth_manager import OpenAIAuthManager
from functools import partialmethod, partial

# once gets importted, prepare to init a manager
GLOBAL_MANAGER = OpenAIAuthManager()

from openai_manager.api_resources import Completion  # avoid parially initialized module error

# Completion = partialmethod()

__all__ = [
    "Completion"
]