from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import asyncio
import aiohttp
import time
import os
import traceback
from openai_manager.utils import logger


# notice loading custom YAML config will overwrite these envvars
GLOBAL_NUM_REQUEST_LIMIT = int(os.getenv(
    "OPENAI_GLOBAL_NUM_REQUEST_LIMIT", 500))
PROMPTS_PER_ASYNC_BATCH = int(os.getenv(
    "OPENAI_PROMPTS_PER_ASYNC_BATCH", 1000))
# 20 requests per minute in `code-davinci-002`, we set it as default
REQUESTS_PER_MIN_LIMIT = int(os.getenv(
    "OPENAI_REQUESTS_PER_MIN_LIMIT", 10))
# 40,000 tokens per minute in `code-davinci-002`, we set it as default
TOKENS_PER_MIN_LIMIT = int(os.getenv("TOKENS_PER_MIN_LIMIT", 40_000))


@dataclass
class StatusTracker:
    # from: https://github.com/openai/openai-cookbook/blob/2f5e350bbe66a418184899b0e12f182dbb46a156/examples/api_request_parallel_processor.py
    """
    Stores metadata about each auth's progress.
    Will trigger a cool-off period if rate limits are hit by asyncio.sleep()
    """
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    # used to cool off after hitting rate limits
    time_of_last_rate_limit_error: int = 0


@dataclass
class RateLimitTracker:
    """
    Keep track of each auth's rate limit status.
    OpenAI applys rate limit in both `requests per minute` and `tokens per minute`.
    RateLimit Tracker should use **second-level** tracker for requests in practice, 
    as OpenAI told in https://platform.openai.com/docs/guides/rate-limits/overview
    """
    next_request_time: int = 0
    seconds_to_pause_after_rate_limit_error: int = 10
    seconds_to_sleep_each_loop: int = 0.001
    available_request_capacity: int = REQUESTS_PER_MIN_LIMIT
    available_token_capacity: int = TOKENS_PER_MIN_LIMIT
    # epsilon_time: float = 0.001  # 1ms delay for corotine rotation
    last_update_time: int = time.time()

    def __post_init__(self):
        # a request at T, next request should be at T + (60 / available_request_capacity)
        self.seconds_per_request = 60 / self.available_request_capacity
        # X tokens at T, next request should be at T + (60 / available_token_capacity) * X
        self.seconds_per_token = 60 / self.available_token_capacity


@dataclass
class OpenAIAuth:
    auth_index: int
    api_key: str
    status_tracker: StatusTracker = field(default_factory=StatusTracker)
    ratelimit_tracker: RateLimitTracker = field(
        default_factory=RateLimitTracker)
    proxy: Optional[str] = None
    is_okay: bool = True
    in_use: bool = False


def task_id_generator_function():
    i = 0
    while True:
        yield i
        i += 1


async def create_session():
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=GLOBAL_NUM_REQUEST_LIMIT)
    )


class OpenAIAuthManager:
    # when importing openai_manager, this class will be initialized
    endpoints = {
        'completions': 'https://api.openai.com/v1/completions',
        'embeddings': 'https://api.openai.com/v1/embeddings',
        'chat_completions': 'https://api.openai.com/v1/chat/completions'
    }
    task_id_generator = task_id_generator_function()

    def __init__(self) -> None:
        # ".env" overrides envvars
        env_source = self.parse_dict_for_env_file() if os.path.exists('.env') else os.environ
        self.auths = self.build_auth_from_envvars(dict_source=env_source)
        logger.warning(f"Loaded {len(self.auths)} OpenAI auths...")
        # self.session = asyncio.get_event_loop().run_until_complete(create_session())
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=GLOBAL_NUM_REQUEST_LIMIT)
        )
        # DeprecationWarning: session was shared among corotinues, allowing keeping connection alive

    def parse_dict_for_env_file(self) -> Dict[str, str]:
        env_dict = dict()
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                env_lines = f.readlines()
                env_lines = [line.split('=') for line in env_lines]
                for env_line in env_lines:
                    if len(env_line) != 2:
                        # skip invalid lines
                        continue
                    env_dict[env_line[0].strip()] = env_line[1].strip()
        return env_dict

    def build_auth_from_envvars(self, dict_source) -> List[OpenAIAuth]:
        # read from config file and build auths
        # auth priority: .env > env var > config file;
        # return a list of auths
        auths = []
        auth_i = 0
        default_proxy = dict_source.get('OPENAI_API_PROXY', None)
        if not default_proxy:
            default_proxy = None
        else:
            logger.warning(f'default proxy is set to {default_proxy}')
        for env_key, env_value in dict_source.items():
            if env_key.startswith('OPENAI_API_KEY'):
                env_key = env_key[len('OPENAI_API_KEY'):]
                auths.append(OpenAIAuth(auth_index=auth_i,
                                        api_key=env_value,
                                        proxy=dict_source.get(
                                            f'OPENAI_API_PROXY{env_key}', default_proxy)  # global proxy overwrites all if not provided proxy
                                        ))
                auth_i += 1
        return auths

    def append_auth_from_config(self, config_path: Optional[str] = None, config_dicts: Optional[List[Dict[str, Any]]] = None):
        if config_path is not None:
            try:
                import yaml
            except ImportError:
                logger.warning(
                    f"pyyaml is not installed, run `pip install pyyaml` in your current environment.")
                return
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            for key, value in config.items():
                if key.startswith('auth'):
                    requests_per_min = value.pop(
                        'requests_per_min', REQUESTS_PER_MIN_LIMIT)
                    tokens_per_min = value.pop(
                        'tokens_per_min', TOKENS_PER_MIN_LIMIT)
                    self.auths.append(OpenAIAuth(auth_index=len(self.auths),
                                                 ratelimit_tracker=RateLimitTracker(
                        available_request_capacity=requests_per_min,
                        available_token_capacity=tokens_per_min),
                        **value
                    ))
        if config_dicts is not None:
            # madantory keys `api_key` / `proxy`, etc...
            self.auths.extend([OpenAIAuth(auth_index=i + len(self.auths), **config_dict)
                              for i, config_dict in enumerate(config_dicts)])

    def get_available_auth(self) -> OpenAIAuth:
        # return the first available auth
        for auth in self.auths:
            if auth.is_okay and not auth.in_use:
                auth.in_use = True
                return auth

    def __del__(self):
        # close the session to avoid aiohttp error
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.session.close())


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []

    async def call_API_pure(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        auth: OpenAIAuth,
        session: aiohttp.ClientSession,
        thread_id: int,
    ):
        # `pure version` remove ratelimit checker and capacity updater
        logger.info(f"thread {thread_id} Starting request #{self.task_id}")
        error = None
        try:
            post_params = {
                'url': request_url,
                'headers': request_header,
                'json': self.request_json
            }
            if auth.proxy:
                post_params.update({'proxy': auth.proxy})

            async with session.post(**post_params) as response:
                # most request gets blocked here; so our restriction is not useful
                response = await response.json()
            if "error" in response:
                # TODO: 1. disable auth if auth is invalid
                # TODO: 2. put into retry_queue if it is a temporary error
                # TODO: 3. throw exceptions immediately if some permanent errors occur
                logger.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                auth.status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    logger.debug(
                        f"Set time_of_last_rate_limit_error to {time.time()}")
                    # each time when triggering rate limit, record capacity:
                    # logger.warning(
                    #     f"Rate limit triggered: capacity: {auth.ratelimit_tracker.available_request_capacity} (requests) / {auth.ratelimit_tracker.available_token_capacity} (tokens)")
                    logger.warning(
                        f'Rate limit triggered: thread {thread_id} auth {auth.auth_index} now {time.time()} next_request_time {auth.ratelimit_tracker.next_request_time}'
                    )
                    auth.status_tracker.time_of_last_rate_limit_error = time.time()
                    auth.status_tracker.num_rate_limit_errors += 1
                    # rate limit errors are counted separately
                    auth.status_tracker.num_api_errors -= 1

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logger.warning(
                f"Request {self.task_id} failed with Exception {e}")
            traceback.print_exc()
            auth.status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            retry_queue.put_nowait(self)
            if self.attempts_left == 0:
                logger.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                auth.status_tracker.num_tasks_in_progress -= 1
                auth.status_tracker.num_tasks_failed += 1
        else:
            auth.status_tracker.num_tasks_in_progress -= 1
            auth.status_tracker.num_tasks_succeeded += 1

        return {'response': response, 'error': error, 'task_id': self.task_id}
