from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import asyncio
import aiohttp
import time
import math
import os
import traceback
from tqdm.asyncio import tqdm_asyncio
from openai_manager.utils import num_tokens_consumed_from_request, deprecated, logger

# exception import
from openai_manager.exceptions import NoAvailableAuthException


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


@deprecated
async def give_up_task(reason: str, task_id: int):
    return {"error": reason, "task_id": task_id}


@deprecated
async def batch_submission(auth_manager: OpenAIAuthManager, prompts: List[str], return_openai_response=False, no_tqdm=False, **kwargs) -> List[Dict[str, Any]]:
    """
    This batch_submission is deprecated! Use the one in `producer_consumer.py`!
    """
    # pre-check: empty auth list
    if len(auth_manager.auths) == 0:
        raise NoAvailableAuthException(f"No available OpenAI Auths!")
    ret = []
    async_gather = asyncio.gather if no_tqdm else tqdm_asyncio.gather

    retry_queue = asyncio.Queue(maxsize=min(
        len(prompts), PROMPTS_PER_ASYNC_BATCH))  # an asynchronized queue;
    # retry_queue = Queue(maxsize=len(prompts))  # a synchronized queue; don't know if it is usable in async task
    api_requests = []
    for prompt in prompts:
        # all unused kwargs goes here!
        request_json = {"prompt": prompt, **kwargs}
        token_consumption = num_tokens_consumed_from_request(
            request_json=request_json,
            api_endpoint='completions',
            token_encoding_name='cl100k_base',
        )
        api_requests.append(APIRequest(
            task_id=next(auth_manager.task_id_generator),
            request_json=request_json,
            token_consumption=token_consumption,
            attempts_left=3,
        ))
    # begin to submit tasks using different OpenAIAuth
    num_async_steps = math.ceil(len(prompts) / PROMPTS_PER_ASYNC_BATCH)
    for i in range(num_async_steps):
        task_lst = []
        async_batch = api_requests[i *
                                   PROMPTS_PER_ASYNC_BATCH: (i + 1) * PROMPTS_PER_ASYNC_BATCH]
        auth_i = 0
        for async_sample in async_batch:
            # current setting is to create_task first, check ratelimit in `call_API`
            task = asyncio.create_task(
                async_sample.call_API(
                    request_url=auth_manager.endpoints['completions'],
                    request_header={
                        'Authorization': f'Bearer {auth_manager.auths[auth_i].api_key}'},
                    retry_queue=retry_queue,
                    status_tracker=auth_manager.auths[auth_i].status_tracker,
                    ratelimit_tracker=auth_manager.auths[auth_i].ratelimit_tracker,
                    session=auth_manager.session,
                )
            )
            task_lst.append(task)
            auth_i = (auth_i + 1) % len(auth_manager.auths)  # auth rotation
            # TODO: allow balancing auths by weights
        # task_lst completed, gatehr results
        results = await async_gather(*task_lst)  # should await all results!
        task_lst = []  # empty task_lst for retry usage...
        # gathered results sometimes have some invalid response,
        # recorded in APIRequest.result and added to retry_queue
        # we now rerun the retry_queue until it is empty

        # I don't think the retry strategy is robust, since gets stuck here.
        while not retry_queue.empty():
            # Note that some requests might have natural failures,
            # e.g. exceed token limitation, so in APIRequest.call_API, we should exclude such cases in retry_queue
            # TODO: exclude wrong instances in retry_queue
            retry_request = retry_queue.get_nowait()
            retry_request.attempts_left -= 1
            if retry_request.attempts_left > 0:
                task = asyncio.create_task(
                    retry_request.call_API(
                        request_url=auth_manager.endpoints['completions'],
                        request_header={
                            'Authorization': f'Bearer {auth_manager.auths[auth_i].api_key}'},
                        retry_queue=retry_queue,
                        status_tracker=auth_manager.auths[auth_i].status_tracker,
                        ratelimit_tracker=auth_manager.auths[auth_i].ratelimit_tracker,
                        session=auth_manager.session,
                    )
                )
                task_lst.append(task)
                auth_i = (auth_i + 1) % len(auth_manager.auths)
            else:
                # a request has been retried 3 times, we give up
                task_lst.append(asyncio.create_task(
                    give_up_task("Too many attempts", task_id=retry_request.task_id))
                )
        # gather again for retry_queue
        retry_results = await async_gather(*task_lst)
        # match retry_results back to original order by overwriting results
        for retry_result in retry_results:
            k = 0
            while k < len(results):  # run scanning since task_id can't be used as index
                if results[k]['task_id'] == retry_result['task_id']:
                    results[k] = retry_result
                    break
                k += 1
        ret.extend(results)
    return ret


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
        # status_tracker: StatusTracker,
        # ratelimit_tracker: RateLimitTracker,
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
            # if "Rate limit" in response["error"].get("message", ""):
            #     self.attempts_left += 1  # do not consider rate limit
            # when next time it gets popped out of queue, will raise Exception
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


@deprecated
def sync_batch_submission(auth_manager, prompt, debug=False, **kwargs):
    loop = asyncio.get_event_loop()
    responses_with_error_and_task_id = loop.run_until_complete(
        batch_submission(auth_manager, prompt, **kwargs))
    return [response['response'] for response in responses_with_error_and_task_id]
