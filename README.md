## OpenAI-Manager

![pypi](https://img.shields.io/pypi/v/openai-manager.svg)
![versions](https://img.shields.io/pypi/pyversions/openai-manager.svg)

Speed up your OpenAI requests by balancing prompts to multiple API keys. Quite useful if you are playing with `code-davinci-002` endpoint.

### Disclaimer

Before using this tool, you are required to read the EULA and ToS of OpenAI L.P. carefully. Actions that violate the OpenAI user agreement may result in the API Key and associated account being suspended. The author shall not be held liable for any consequential damages.

### Design
TL;DR: this package helps you manage rate limit (both request-level and token-level) for each api_key for maximum number of requests to OpenAI API.

This is extremely helpful if you use `CODEX` endpoint or you have a handful of free-trial accounts due to limited budget. Free-trial accounts apply **strict** rate limit.

### Quickstart

1. Install openai-manager on PyPI. Notice we need Python 3.7+ for maximum compatibility of `asyncio`.
   ```bash
   pip install openai-manager
   ```

2. Prepare your OpenAI credentials in 
   1. Environmental Varibles: any envvars beginning with `OPENAI_API_KEY` will be used to initialized the manager. Best practice to load your api keys is to prepare a `.env` file like: 
   
   ```bash
   OPENAI_API_KEY_1=sk-Nxo******
   OPENAI_API_KEY_2=sk-TG2******
   OPENAI_API_KEY_3=sk-Kpt******
   # You can set a global proxy for all api_keys
   OPENAI_API_PROXY=http://127.0.0.1:7890
   # You can also append proxy to each api_key. 
   # Make sure the indices match.
   OPENAI_API_PROXY_1=http://127.0.0.1:7890
   OPENAI_API_PROXY_2=http://127.0.0.1:7890
   OPENAI_API_PROXY_3=http://127.0.0.1:7890
   ```

   Then load your environmental varibles before running any scripts:
   ```bash
   export $(grep -v '^#' .env | xargs)
   ```

   2. YAML config file: you can add more fine-grained restrictions on each API key if you know the ratelimit for each key in advance. See [example_config.yml](/example_config.yml) for details.
   ```python
   import openai_manager
   openai_manager.append_auth_from_config(config_path='example_config.yml')
   ```

3. Two ways to use `openai_manager`:
   1. Use it just like how you use official `openai` package. We implement exact the same call signature as official `openai` package.
        ```python
        import openai as official_openai
        import openai_manager
        from openai_manager.utils import timeit
        
        @timeit
        def test_official_separate():
            for i in range(10):
                prompt = "Once upon a time, "
                response = official_openai.Completion.create(
                    model="code-davinci-002",
                    prompt=prompt,
                    max_tokens=20,
                )
                print("Answer {}: {}".format(i, response["choices"][0]["text"]))

        @timeit
        def test_manager():
            prompt = "Once upon a time, "
            prompts = [prompt] * 10
            responses = openai_manager.Completion.create(
                model="code-davinci-002",
                prompt=prompts,
                max_tokens=20,
            )
            assert len(responses) == 10
            for i, response in enumerate(responses):
                print("Answer {}: {}".format(i, response["choices"][0]["text"]))
        ```
    2. Use it as a proxy server between you and OpenAI endpoint. First, run `python -m openai_manager.serving --port 8000 --host localhost --api_key [your custom key]`. Then set up the official python `openai` package:
        ```python
        import openai
        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "[your custom key]"

        # run like normal
        prompt = ["Once upon a time, "] * 10
        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt,
            max_tokens=20,
        )
        print(response["choices"][0]["text"])
        ```

### Configuration

Most configurations are manupulated by environmental variables. 

- `GLOBAL_NUM_REQUEST_LIMIT`: aiohttp connection limit, default is `500`
- `REQUESTS_PER_MIN_LIMIT`: number of requests per minute, default is `10`; config file will overwrite this
- `TOKENS_PER_MIN_LIMIT`: number of tokens per minute, default is `40000`; config file will overwrite this
- `COROTINE_PER_AUTH`: number of corotine per api_key, default is `3`; decrease it to 1 if ratelimit errors are triggered too often
- `ATTEMPTS_PER_PROMPT`: number of attempts per prompt, default is `5`
- `RATELIMIT_AFTER_SUBMISSION`: whether to track ratelimit after submission, default is `True`; keep it enabled if response takes a long time
- `OPENAI_LOG_LEVEL`: default log level is WARNING, 10-DEBUG, 20-INFO, 30-WARNING, 40-ERROR, 50-CRITICAL; set to 10 if you are getting stuck and want to do some diagnose

Rate limit triggers will be visible under `logging.WARNING`. Run `export OPENAI_LOG_LEVEL=40` to ignore rate limit warnings if you believe current setting is stable enought.

### Performance Assessment

After ChatCompletion release, the `code-davinci-002` endpoint becomes slow. Using 10 API keys, running 100 completions with `max_tokens=20` and other hyperparameters left default took 90 seconds on average. Using official API, it took 10 seconds per completion, thus 1000 in total. 

Theroticallly, the throughput increases linearly with the number of API keys. 

### Frequently Asked Questions

1. Q: Why don't we just use official batching function?

   ```python
    prompt = "Once upon a time, "
    prompts = [prompt] * 10
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompts,  # official batching allows multiple prompts in one request
        max_tokens=20,
    )
    assert len(response["choices"]) == 10
    for i, answer in enumerate(response["choices"]):
        print("Answer {}: {}".format(i, answer["text"]))
   ```
   
   A: Some OpenAI endpoints (like `code-davinci-002`) apply strict token-level rate limit, even if you upgrade to pay-as-you-go user. Simple batching would not solve this.

### Acknowledgement

- [openai-cookbook](https://github.com/openai/openai-cookbook): Best practice when dealing with official APIs.
- [openai-python](https://github.com/openai/openai-python): Official Python version of OpenAI.

### TODO

- [ ] Support all functions in OpenAI Python API.
  - [x] Completions
  - [x] Embeddings
  - [ ] Generations
  - [x] ChatCompletions
- [x] Better back-off strategy for maximum throughput.
- [x] Properly handling exceptions raised by OpenAI API.
- [ ] Automatic batching prompts to reduce the number of requests.
- [ ] Automatic rotation of tons of OpenAI API Keys. (Removing invaild, adding new, etc.)
- [x] Serving as a reverse proxy to balance official requests.


### Donation

If this package helps your research, consider making a donation via GitHub! 