import asyncio
from typing import List, Dict, Any, Optional
from openai_manager.auth_manager import OpenAIAuthManager, OpenAIAuth, APIRequest, REQUESTS_PER_MIN_LIMIT, TOKENS_PER_MIN_LIMIT
import os
import time
from openai_manager.exceptions import NoAvailableAuthException, AttemptsExhaustedException
from openai_manager.utils import num_tokens_consumed_from_request, logger, str2bool
from tqdm import tqdm

# notice loading custom YAML config will overwrite these envvars

PROMPTS_PER_ASYNC_BATCH = int(
    os.getenv("OPENAI_PROMPTS_PER_ASYNC_BATCH", 1000))
COROTINE_PER_AUTH = int(os.getenv("COROTINE_PER_AUTH", 3))
ATTEMPTS_PER_PROMPT = int(os.getenv("ATTEMPTS_PER_PROMPT", 5))
RATELIMIT_AFTER_SUBMISSION = str2bool(
    os.getenv("RATELIMIT_AFTER_SUBMISSION", "True"))

if COROTINE_PER_AUTH == 1:
    RATELIMIT_AFTER_SUBMISSION = False


async def consume_submit(q: asyncio.Queue, auth: OpenAIAuth, results: List[dict], auth_manager, thread_id: int, pbar: Optional[tqdm] = None):
    while True:
        logger.debug(
            f"thread {thread_id} auth {auth.auth_index} waiting for submission...")
        item = await q.get()
        item: APIRequest
        logger.debug(
            f"thread {thread_id} auth {auth.auth_index} received submission {item}")
        # check if exhausted attempts
        if item.attempts_left == 0:
            raise AttemptsExhaustedException(
                f"Request {item} exhausted all attempts.")

        item.attempts_left -= 1

        # --- below are our second-level rate limit controller, should be more strict! ---
        before_submission_time = time.time()
        if auth.ratelimit_tracker.next_request_time != 0:
            auth.ratelimit_tracker.next_request_time = max(
                auth.ratelimit_tracker.next_request_time +
                auth.ratelimit_tracker.seconds_per_request,
                auth.ratelimit_tracker.next_request_time +
                auth.ratelimit_tracker.seconds_per_token * item.token_consumption
            )
        else:
            auth.ratelimit_tracker.next_request_time = before_submission_time  # first request
        logger.info(
            f"thread {thread_id} auth {auth.auth_index} update next_request_time to {auth.ratelimit_tracker.next_request_time}")
        if before_submission_time < auth.ratelimit_tracker.next_request_time:
            logger.info(
                f"thread {thread_id} auth {auth.auth_index} waiting for next request time..., before_submission_time: {before_submission_time}, next_request_time: {auth.ratelimit_tracker.next_request_time}")
            await asyncio.sleep(auth.ratelimit_tracker.next_request_time - before_submission_time)

        response = await item.call_API_pure(
            request_url=auth_manager.endpoints['completions'],
            request_header={
                'Authorization': f'Bearer {auth.api_key}'},
            retry_queue=q,
            auth=auth,
            session=auth_manager.session,
            thread_id=thread_id,
        )
        # we update next_request_time when receiving the response, as sometimes response takes very long
        if RATELIMIT_AFTER_SUBMISSION:
            after_submission_time = time.time()
            auth.ratelimit_tracker.next_request_time = max(
                after_submission_time +
                auth.ratelimit_tracker.seconds_per_request,
                after_submission_time +
                auth.ratelimit_tracker.seconds_per_token * item.token_consumption
            )
            logger.info(
                f"after submission, thread {thread_id} auth {auth.auth_index} update next_request_time to {auth.ratelimit_tracker.next_request_time}")
            logger.info(
                f"thread {thread_id} auth {auth.auth_index} received request {item.task_id} at {time.time()}, next_request_time is {auth.ratelimit_tracker.next_request_time}")

        if pbar is not None:
            pbar.update()
            if response['error']:
                pbar.total += 1   # put error back to queue
                pbar.refresh()

        if response['error']:
            logger.warning(
                f"thread {thread_id} auth {auth.auth_index} experience error...")

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
        # TODO: automatic batching for a large number of prompts
        token_consumption = num_tokens_consumed_from_request(
            request_json=request_json,
            api_endpoint='completions',
            token_encoding_name='cl100k_base',
        )
        await q.put(APIRequest(
            task_id=next(auth_manager.task_id_generator),
            request_json=request_json,
            token_consumption=token_consumption,
            attempts_left=ATTEMPTS_PER_PROMPT,
        ))

    consumers = []
    thread_id = 0
    for auth in auth_manager.auths:
        consumers.extend([asyncio.create_task(consume_submit(
            q, auth, ret, auth_manager, thread_id + j, pbar)) for j in range(COROTINE_PER_AUTH)])
        thread_id += COROTINE_PER_AUTH
    await q.join()  # wait until queue is empty
    for c in consumers:  # shutdown all consumers
        c.cancel()
    # gather results, ignore errors
    return list(sorted([result for result in ret if not result['error']], key=lambda x: x['task_id']))


def sync_batch_submission(auth_manager, prompt, debug=False, no_tqdm=False, **kwargs):
    loop = asyncio.get_event_loop()
    pbar = tqdm(total=len(prompt)) if not no_tqdm else None
    responses_with_error_and_task_id = loop.run_until_complete(
        batch_submission(auth_manager, prompt, pbar=pbar, **kwargs))
    if not pbar:
        pbar.close()
    # keep 'response' only
    # TODO: if gets automatic batching, remember to restore them back
    return [response['response'] for response in responses_with_error_and_task_id]


if __name__ == '__main__':
    auth_manager = OpenAIAuthManager()
    print(sync_batch_submission(auth_manager, [
          "Hello world!"] * 30, max_tokens=20, model='code-davinci-002'))
