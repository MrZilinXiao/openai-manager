# light-weight API wrapper for identical call signatures with official openai-python
# from openai_manager.auth_manager import sync_batch_submission
from openai_manager.producer_consumer import sync_batch_submission
from openai_manager import GLOBAL_MANAGER


class Completion:
    @classmethod
    def create(cls, **params):
        # print(params)
        return sync_batch_submission(GLOBAL_MANAGER, **params)


class Embedding:
    pass
