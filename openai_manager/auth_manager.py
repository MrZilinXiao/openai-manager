from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import asyncio
from queue import Queue
import aiohttp
import logging
import time
import math
import os
from tqdm.asyncio import tqdm_asyncio
from openai_manager.utils import num_tokens_consumed_from_request

logging.basicConfig(level=os.getenv("OPENAI_LOG_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)
logger.debug(f"Logger level: {logger.level}")

# notice loading custom YAML config will overwrite these envvars
GLOBAL_NUM_REQUEST_LIMIT = os.getenv("OPENAI_GLOBAL_NUM_REQUEST_LIMIT", 10)
PROMPTS_PER_ASYNC_BATCH = os.getenv("OPENAI_PROMPTS_PER_ASYNC_BATCH", 1000)
REQUESTS_PER_MIN_LIMIT = os.getenv("OPENAI_REQUESTS_PER_MIN_LIMIT", 20)  # 20 requests per minute in `code-davinci-002`
TOKENS_PER_MIN_LIMIT = os.getenv("TOKENS_PER_MIN_LIMIT", 40_000)  # 40,000 tokens per minute in `code-davinci-002`

@dataclass
class StatusTracker:
    # from: https://github.com/openai/openai-cookbook/blob/2f5e350bbe66a418184899b0e12f182dbb46a156/examples/api_request_parallel_processor.py
    """
    Stores metadata about each auth's progress.
    Will trigger a cool-off period if rate limits are hit by asyncio.sleep()
    Remember to check quota before running any API calls!
    """
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

@dataclass
class RateLimitTracker:
    """
    Keep track of each auth's rate limit status.
    OpenAI applys rate limit in both `requests per minute` and `tokens per minute`.
    """
    next_request_time: int = 0
    seconds_to_pause_after_rate_limit_error: int = 15
    seconds_to_sleep_each_loop: int = 0.001
    available_request_capacity: int = REQUESTS_PER_MIN_LIMIT
    available_token_capacity: int = TOKENS_PER_MIN_LIMIT
    last_update_time: int = time.time()
    

@dataclass
class OpenAIAuth:
    # auth_index: int
    api_key: str
    status_tracker: StatusTracker = StatusTracker()
    ratelimit_tracker: RateLimitTracker = RateLimitTracker()
    proxy: Optional[str] = None
    is_okay: bool = True
    in_use: bool = False


class OpenAIAuthManager:
    # when importing openai_manager, this class will be initialized
    endpoints = {
        'completions': 'https://api.openai.com/v1/completions', 
    }
    def __init__(self) -> None:
        # dummy auth here; 
        # as we use corotine, we do not need to care about race condition
        self.auths = self.build_auth_from_env()
        logger.info(f"Loaded {len(self.auths)} OpenAI auths...")
        # [{'some_key': 'some_proxy_or_None'}, ...]
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=GLOBAL_NUM_REQUEST_LIMIT)
        )  # DeprecationWarning: session was shared among corotinues, allowing keeping connection alive

    def build_auth_from_env(self) -> List[OpenAIAuth]:
        # read from config file and build auths
        # auth priority: env var > config file; allowing both but not recommended
        # return a list of auths
        auths = []
        for env_key, env_value in os.environ.items():
            if env_key.startswith('OPENAI_API_KEY'):
                env_key = env_key[len('OPENAI_API_KEY'): ]
                auths.append(OpenAIAuth(api_key=env_value, 
                                        proxy=os.getenv(f'OPENAI_API_PROXY{env_key}', None)
                                        ))
        return auths
                
    def append_auth_from_config(self, config_path: Optional[str] = None, config_dicts: Optional[List[Dict[str, Any]]] = None):
        if config_path is not None:
            pass
        if config_dicts is not None:
            # madantory keys `api_key` / `proxy`, etc...
            self.auths.extend([OpenAIAuth(**config_dict) for config_dict in config_dicts])
        

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

# auth_manager = OpenAIAuthManager()  # global once imported openai_manager

def task_id_generator():
    i = 0
    while True:
        yield i
        i += 1

async def give_up_task(reason: str, task_id: int):
    return {"error": reason, "task_id": task_id}


async def batch_submission(auth_manager: OpenAIAuthManager, prompts: List[str], return_openai_response=False, no_tqdm=False, **kwargs) -> List[Dict[str, Any]]:
    """
    auth_manager is provided globally
    """
    ret = []
    async_gather = asyncio.gather if no_tqdm else tqdm_asyncio.gather

    retry_queue = asyncio.Queue(maxsize=len(prompts))  # an asynchronized queue; 
    # retry_queue = Queue(maxsize=len(prompts))  # a synchronized queue; don't know if it is usable in async task
    api_requests = []
    for prompt in prompts:
        request_json = {"prompt": prompt, **kwargs}  # all unused kwargs goes here!
        token_consumption = num_tokens_consumed_from_request(
            request_json=request_json, 
            api_endpoint='completions',
            token_encoding_name='cl100k_base',
        )
        api_requests.append(APIRequest(
            task_id=next(task_id_generator()),
            request_json=request_json,
            token_consumption=token_consumption,
            attempts_left=3,
        ))
    # begin to submit tasks using different OpenAIAuth
    num_async_steps = math.ceil(len(prompts) / PROMPTS_PER_ASYNC_BATCH)
    for i in range(num_async_steps):
        task_lst = []
        async_batch = api_requests[i * PROMPTS_PER_ASYNC_BATCH: (i + 1) * PROMPTS_PER_ASYNC_BATCH]
        auth_i = 0
        for async_sample in async_batch:
            # current setting is to create_task first, check ratelimit in `call_API`
            task = asyncio.create_task(
                async_sample.call_API(
                    request_url=auth_manager.endpoints['completions'],
                    request_header={'Authorization': f'Bearer {auth_manager.auths[auth_i].api_key}'},
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
                    request_header={'Authorization': f'Bearer {auth_manager.auths[auth_i].api_key}'},
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

    async def call_API(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        ratelimit_tracker: RateLimitTracker,
        session: aiohttp.ClientSession,
    ):
        """Calls the OpenAI API and saves results."""
        # before calling API, check
        # 1. if we have enough capacity
        while ratelimit_tracker.available_request_capacity < 1 or \
            ratelimit_tracker.available_token_capacity < self.token_consumption:
            asyncio.sleep(0.1)  # wait until capacity fulfilled
            # This is not the best practice, maybe we should seek producer-consumer model
            # which involves a main loop to manage the batch...
        # 2. if we hit rate limit recently
        seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
        if seconds_since_rate_limit_error < ratelimit_tracker.seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = ratelimit_tracker.seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            asyncio.sleep(remaining_seconds_to_pause)
            logger.warn(f"Pausing to cool down for {remaining_seconds_to_pause:.4f} seconds. If you see this often, consider lower your rate limit configuration.")

        logger.info(f"Starting request #{self.task_id}")
        error = None
        try:
            # async with aiohttp.ClientSession() as session:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                # TODO: 1. disable auth if auth is invalid
                # TODO: 2. put into retry_queue if it is a temporary error
                # TODO: 3. throw exceptions immediately if some permanent errors occur
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                # append_to_jsonl([self.request_json, self.result], save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            # append_to_jsonl([self.request_json, response], save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            # logging.debug(f"Request {self.task_id} saved to {save_filepath}")
        
        # before return, update available token capacity and available request capacity
        current_time = time.time()
        seconds_since_update = current_time - ratelimit_tracker.last_update_time
        ratelimit_tracker.available_token_capacity = min(
            ratelimit_tracker.available_token_capacity + TOKENS_PER_MIN_LIMIT * seconds_since_update / 60,
            TOKENS_PER_MIN_LIMIT
        )
        ratelimit_tracker.available_request_capacity = min(
            ratelimit_tracker.available_request_capacity + REQUESTS_PER_MIN_LIMIT * seconds_since_update / 60,
            REQUESTS_PER_MIN_LIMIT
        )
        ratelimit_tracker.last_update_time = current_time

        # print({'response': response, 'error': error, 'task_id': self.task_id})
        return {'response': response, 'error': error, 'task_id': self.task_id}

def sync_batch_submission(auth_manager, prompt, debug=False, **kwargs):
    loop = asyncio.get_event_loop()
    responses_with_error_and_task_id = loop.run_until_complete(batch_submission(auth_manager, prompt, **kwargs))
    return [response['response'] for response in responses_with_error_and_task_id]

if __name__ == '__main__':
    # simple test
    prompts = ['Testing OpenAI!!' for _ in range(100)]
    print(sync_batch_submission(prompts))