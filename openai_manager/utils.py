import tiktoken
import warnings
import functools
import os
import logging
import argparse
import time

logging.basicConfig(level=int(os.getenv("OPENAI_LOG_LEVEL", logging.WARNING)))
logger = logging.getLogger(__name__)
logger.debug(f"Logger level: {logger.level}")


def timeit(func):
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


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
        num_tokens = 0
        # note: future models may deviate from this
        if request_json['model'] == "gpt-3.5-turbo-0301":
            # chat completions are also in list
            for messages_lst in request_json['messages']:
                for message in messages_lst:
                    # every message follows <im_start>{role/name}\n{content}<im_end>\n
                    num_tokens += 4
                    for key, value in message.items():
                        num_tokens += len(encoding.encode(value))
                        if key == "name":  # if there's a name, the role is omitted
                            num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script')
