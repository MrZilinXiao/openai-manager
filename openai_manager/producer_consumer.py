import asyncio
from typing import List, Dict, Any
from openai_manager.auth_manager import OpenAIAuthManager, OpenAIAuth, APIRequest
import logging
import os
import time
from openai_manager.exceptions import NoAvailableAuthException, AttemptsExhaustedException
from openai_manager.utils import num_tokens_consumed_from_request
from tqdm import tqdm

logging.basicConfig(level=int(os.getenv("OPENAI_LOG_LEVEL", logging.WARNING)))
logger = logging.getLogger(__name__)
logger.debug(f"Logger level: {logger.level}")

# notice loading custom YAML config will overwrite these envvars
GLOBAL_NUM_REQUEST_LIMIT = os.getenv("OPENAI_GLOBAL_NUM_REQUEST_LIMIT", 10)
PROMPTS_PER_ASYNC_BATCH = os.getenv("OPENAI_PROMPTS_PER_ASYNC_BATCH", 1000)
# 20 requests per minute in `code-davinci-002`
REQUESTS_PER_MIN_LIMIT = os.getenv("OPENAI_REQUESTS_PER_MIN_LIMIT", 10)
# 40,000 tokens per minute in `code-davinci-002`
TOKENS_PER_MIN_LIMIT = os.getenv("TOKENS_PER_MIN_LIMIT", 20_000)


async def consume_submit(q: asyncio.Queue, auth: OpenAIAuth, results: List[dict], auth_manager, thread_id: int, pbar=None):
    while True:
        logger.debug(f"thread {thread_id} waiting for submission...")
        item = await q.get()
        item: APIRequest
        logger.debug(f"thread {thread_id} received submission {item}")
        # check if exhausted attempts
        if item.attempts_left == 0:
            raise AttemptsExhaustedException(
                f"Request {item} exhausted all attempts.")
        # check if current auth is rate limited
        while auth.ratelimit_tracker.available_request_capacity < 1 or \
                auth.ratelimit_tracker.available_token_capacity < item.token_consumption:
            # logger.debug(
            #     f"Waiting for capacity to be available at {time.time()}...")
            await asyncio.sleep(0.1)  # wait until capacity fulfilled

        auth.ratelimit_tracker.available_request_capacity -= 1
        auth.ratelimit_tracker.available_token_capacity -= item.token_consumption
        item.attempts_left -= 1

        seconds_since_rate_limit_error = time.time(
        ) - auth.status_tracker.time_of_last_rate_limit_error

        if seconds_since_rate_limit_error < auth.ratelimit_tracker.seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = auth.ratelimit_tracker.seconds_to_pause_after_rate_limit_error - \
                seconds_since_rate_limit_error
            logger.warn(
                f"Pausing to cool down for {remaining_seconds_to_pause:.4f} seconds. If you see this often, consider lower your rate limit configuration.")
            await asyncio.sleep(remaining_seconds_to_pause)

        logger.debug(f"thread {thread_id} submitting request... "
                     f"remaining token capacity: {auth.ratelimit_tracker.available_token_capacity}; "
                     f"remaining request capacity: {auth.ratelimit_tracker.available_request_capacity}")
        # get response!
        response = await item.call_API_pure(
            request_url=auth_manager.endpoints['completions'],
            request_header={
                'Authorization': f'Bearer {auth.api_key}'},
            retry_queue=q,
            status_tracker=auth.status_tracker,
            ratelimit_tracker=auth.ratelimit_tracker,
            session=auth_manager.session,
            thread_id=thread_id,
        )

        # before return, update available token capacity and available request capacity
        current_time = time.time()
        seconds_since_update = current_time - auth.ratelimit_tracker.last_update_time
        auth.ratelimit_tracker.available_token_capacity = min(
            auth.ratelimit_tracker.available_token_capacity +
            TOKENS_PER_MIN_LIMIT * seconds_since_update / 60,
            TOKENS_PER_MIN_LIMIT
        )
        auth.ratelimit_tracker.available_request_capacity = min(
            auth.ratelimit_tracker.available_request_capacity +
            REQUESTS_PER_MIN_LIMIT * seconds_since_update / 60,
            REQUESTS_PER_MIN_LIMIT
        )
        logger.debug(f"Current available token capacity: {auth.ratelimit_tracker.available_token_capacity}, "
                     f"current_available_request_capacity: {auth.ratelimit_tracker.available_request_capacity}")
        auth.ratelimit_tracker.last_update_time = current_time

        if pbar is not None:
            pbar.update()

        results.append(response)  # remember to sort by task_id when returning

        q.task_done()


async def batch_submission(auth_manager: OpenAIAuthManager, prompts: List[str], return_openai_response=False, pbar=None, **kwargs) -> List[Dict[str, Any]]:
    if len(auth_manager.auths) == 0:
        raise NoAvailableAuthException(f"No available OpenAI Auths!")
    ret = []
    q = asyncio.Queue()  # storing APIRequest
    for prompt in prompts:
        # all unused kwargs goes here!
        request_json = {"prompt": prompt, **kwargs}
        token_consumption = num_tokens_consumed_from_request(
            request_json=request_json,
            api_endpoint='completions',
            token_encoding_name='cl100k_base',
        )
        await q.put(APIRequest(
            task_id=next(auth_manager.task_id_generator),
            request_json=request_json,
            token_consumption=token_consumption,
            attempts_left=3,
        ))

    consumers = []
    thread_id = 1
    for auth in auth_manager.auths:
        consumers.extend([asyncio.create_task(consume_submit(
            q, auth, ret, auth_manager, thread_id, pbar)) for _ in range(2)])
        thread_id += 1
    await q.join()  # wait until queue is empty
    for c in consumers:  # shutdown all consumers
        c.cancel()
    # gather results
    return list(sorted(ret, key=lambda x: x['task_id']))


def sync_batch_submission(auth_manager, prompt, debug=False, **kwargs):
    loop = asyncio.get_event_loop()
    with tqdm(total=len(prompt)) as pbar:
        responses_with_error_and_task_id = loop.run_until_complete(
            batch_submission(auth_manager, prompt, pbar=pbar, **kwargs))
    # keep 'response' only
    return [response['response'] for response in responses_with_error_and_task_id]


if __name__ == '__main__':
    auth_manager = OpenAIAuthManager()
    print(sync_batch_submission(auth_manager, [
          "Hello world!"] * 30, max_tokens=20, model='code-davinci-002'))
