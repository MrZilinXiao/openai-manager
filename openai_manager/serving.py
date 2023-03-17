# pylint: skip-file
# this script inits a FastAPI endpoint and an OpenAIManager
# allowing distributing openai requests to multiple keys from all official openai api implementations
import uvicorn
import argparse
from fastapi import FastAPI, Request
from openai_manager import GLOBAL_MANAGER
from openai_manager.producer_consumer import batch_submission

app = FastAPI()
api_key = None

@app.post("/v1/completions")
async def completions(request: Request):
    # validate first
    key = request.headers.get('authorization', None)
    if key != f'Bearer {api_key}':  # covered None case
        return {
            "error": {
            "message": f"Incorrect API key provided: {key}. You should set the key in openai-manager via command.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key"
            }
        }
    request_json = await request.json()
    prompt = request_json.pop('prompt')
    if not isinstance(prompt, list):
        prompt = [prompt]
        
    input_lst = [{"prompt": p, **request_json} for p in prompt]
    responses_with_error_and_task_id = await batch_submission(GLOBAL_MANAGER, input_lst, pbar=None, api_endpoint='completions')
    # keep response only, and format into openai api response
    response = [response['response'] for response in responses_with_error_and_task_id]
    assert len(response) > 0, "Empty response"
    # we keep the first request's `id`, `object`, `created`, `model`; sum over all the usage, and combine the `choices`
    return {
        'id': response[0]['id'],
        'object': response[0]['object'],
        'created': response[0]['created'],
        'model': response[0]['model'],
        'choices': [c for res in response for c in res['choices']],  # `index` kept; our implementation ensures ascending order
        'usage': {
            'prompt_tokens': sum([res['usage']['prompt_tokens'] for res in response]),
            'completion_tokens': sum([res['usage']['completion_tokens'] for res in response]),
            'total_tokens': sum([res['usage']['total_tokens'] for res in response]),
        }
    }
    
@app.post("/v1/completions/embeddings")
async def embeddings(request: Request):
    pass

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--reload", action='store_true', default=False)
    parser.add_argument("--custom_config", type=str, default=None)
    parser.add_argument("--api_key", type=str, default="sk-123")  # this api key is used to do self validation, not for openai requests
    # post args checking
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    GLOBAL_MANAGER.append_auth_from_config(config_path=args.custom_config)
    api_key = args.api_key
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)