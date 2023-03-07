# light-weight API wrapper for identical call signatures with official openai-python
# from openai_manager.auth_manager import sync_batch_submission
from openai_manager.producer_consumer import sync_batch_submission
from openai_manager import GLOBAL_MANAGER


class Completion:
    @classmethod
    def create(cls, **params):
        prompt = params.pop('prompt')  # ensure a required field
        return sync_batch_submission(GLOBAL_MANAGER, prompt=prompt, **params)


class Embedding:
    @classmethod
    def create(cls, **params):
        input = params.pop('input')
        return sync_batch_submission(GLOBAL_MANAGER, input=input, **params)


class ChatCompletion:
    @classmethod
    def create(cls, **params):
        messages = params.pop('messages')
        return sync_batch_submission(GLOBAL_MANAGER, messages=messages, **params)
