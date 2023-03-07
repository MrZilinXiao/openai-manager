import tiktoken
import warnings
import functools
import os
import logging
import argparse

logging.basicConfig(level=int(os.getenv("OPENAI_LOG_LEVEL", logging.WARNING)))
logger = logging.getLogger(__name__)
logger.debug(f"Logger level: {logger.level}")


def str2bool(s):
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint == "completions":
        prompt = request_json["prompt"]
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens
        if isinstance(prompt, str):  # single prompt
            prompt_tokens = len(encoding.encode(prompt))
            num_tokens = prompt_tokens + completion_tokens
            return num_tokens
        elif isinstance(prompt, list):  # multiple prompts
            prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
            num_tokens = prompt_tokens + completion_tokens * len(prompt)
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    # if chat_completions request, tokens = all content input tokens
    elif api_endpoint == 'chat_completions':
        input = request_json["messages"]
        # batched or single input is always a list
        assert isinstance(
            input, list), f"Expecting list of strings for 'messages' field in chat_completions request, got {type(input)}"
        is_multiple = isinstance(input[0], list) if input else False
        #   messages=[
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": "Who won the world series in 2020?"},
        #     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #     {"role": "user", "content": "Where was it played?"}
        # ]
        #  if batched, messages can be a list of `messages`
        if is_multiple:  # multiple inputs
            num_tokens = sum([len(encoding.encode(i['content']))
                             for inp in input for i in inp])
        else:
            num_tokens = sum([len(encoding.encode(inp['content']))
                             for inp in input])
        return num_tokens
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script')
