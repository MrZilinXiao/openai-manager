# light-weight API wrapper for identical call signatures with official openai-python
from openai_manager.producer_consumer import sync_batch_submission
from openai_manager import GLOBAL_MANAGER


class Completion:
    @classmethod
    def create(cls, **params):
        prompt = params.pop('prompt')  # ensure a required field
        if not isinstance(prompt, list):
            prompt = [prompt]
        return sync_batch_submission(GLOBAL_MANAGER, prompt=prompt, **params)


class Embedding:
    @classmethod
    def create(cls, **params):
        input = params.pop('input')
        if not isinstance(input, list):
            input = [input]
        return sync_batch_submission(GLOBAL_MANAGER, input=input, **params)


class ChatCompletion:
    @classmethod
    def create(cls, **params):
        messages = params.pop('messages', None)
        if not isinstance(messages, list):
            raise TypeError(f"messages must be a list of dict, got {type(messages)}")
        # when provided in openai style, wrap it with a list, despite this will not allow parallellization
        if isinstance(messages[0], dict):
            messages = [messages]
        return sync_batch_submission(GLOBAL_MANAGER, messages=messages, **params)
