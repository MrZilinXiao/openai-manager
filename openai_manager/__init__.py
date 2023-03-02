# pylint: skip-file
# avoid parially initialized module error
from openai_manager.auth_manager import OpenAIAuthManager  # noqa
GLOBAL_MANAGER = OpenAIAuthManager()  # noqa


from openai_manager.api_resources import Completion  # noqa
from functools import partialmethod, partial  # noqa


__all__ = [
    "Completion"
]

name = "openai-manager"
