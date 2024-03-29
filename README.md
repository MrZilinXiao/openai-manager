## OpenAI-Manager

![pypi](https://img.shields.io/pypi/v/openai-manager.svg)
![versions](https://img.shields.io/pypi/pyversions/openai-manager.svg)
[![Run Unittest](https://github.com/MrZilinXiao/openai-manager/actions/workflows/unittest.yml/badge.svg)](https://github.com/MrZilinXiao/openai-manager/actions/workflows/unittest.yml)


Speed up your OpenAI requests by balancing prompts to multiple API keys. ~~Quite useful if you are playing with `code-davinci-002` endpoint.~~

> Update on 2023/03/24: OpenAI terminated all `CODEX` endpoint access today. An immediate migration to `gpt-3.5-turbo` or other endpoints is needed to ensure the stability of your service.

### Disclaimer

Before using this tool, you are required to read the EULA and ToS of OpenAI L.P. carefully. Actions that violate the OpenAI user agreement may result in the API Key and associated account being suspended. The author shall not be held liable for any consequential damages.

**Caution**: do not deploy this tool directly in Mainland China, Hong Kong SAR or any other locations where OpenAI disallows API usage. Use `OPENAI_API_PROXY` environmental variable to set a proxy (e.g. Japan) for connectting OpenAI API. Failure to do so will bring quick termination of your account.

### Design

![design](static/openai-manager.svg)

TL;DR: this package helps you manage rate limit (both request-level and token-level) for each api_key for maximum number of requests to OpenAI API.

This is extremely helpful if ~~you use `CODEX` endpoint or~~ you have a handful of **free-trial accounts** due to limited budget. Free-trial accounts apply **strict** rate limit.

### Quickstart

1. Install openai-manager on PyPI. Notice we need Python 3.8+ for maximum compatibility of `asyncio` and `tiktoken`.
   ```bash
   pip install -U openai-manager
   ```

2. Prepare your OpenAI credentials in: 
   <details>
   <summary>Environment Variables</summary>
   Any envvars beginning with `OPENAI_API_KEY` will be used to initialized the manager. Best practice to load your api keys is to prepare a `.env` file like: 
   
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
   
   `openai-manager` will try to read the `.env` file in your current working directory. You can also load environmental varibles manually by:

   ```bash
   export $(grep -v '^#' .env | xargs)
   ```
   </details>

   <details>
   <summary>YAML config file</summary>
   You can add more fine-grained restrictions on each API key if you know the ratelimit for each key in advance. See [example_config.yml](/example_config.yml) for details.

   ```python
   import openai_manager
   openai_manager.append_auth_from_config(config_path='example_config.yml')
   ```

   </details>

3. Two ways to use `openai_manager`:
   - Use it just like how you use official `openai` package. We implement exact the same call signature as official `openai` package.
        ```python
        import openai as official_openai
        import openai_manager
        from openai_manager.utils import timeit
        
        @timeit
        def test_official_separate():
            for i in range(10):
                prompt = "Once upon a time, "
                response = official_openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=20,
                )
                print("Answer {}: {}".format(i, response["choices"][0]["text"]))

        @timeit
        def test_manager():
            prompt = "Once upon a time, "
            prompts = [prompt] * 10
            responses = openai_manager.Completion.create(
                model="text-davinci-003",
                prompt=prompts,
                max_tokens=20,
            )
            assert len(responses) == 10
            for i, response in enumerate(responses):
                print("Answer {}: {}".format(i, response["choices"][0]["text"]))
        ```
   - Use it as a proxy server between you and OpenAI endpoint. First, run `python -m openai_manager.serving --port 8000 --host localhost --api_key [your custom key]`. Then set up the official python `openai` package:
        ```python
        import openai
        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "[your custom key]"

        # run like normal
        prompt = ["Once upon a time, "] * 10
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=20,
        )
        print(response["choices"][0]["text"])
        ```

### Configuration

Most configurations are manupulated by environmental variables. 

- `GLOBAL_NUM_REQUEST_LIMIT`: aiohttp connection limit, default is `500`;
- `REQUESTS_PER_MIN_LIMIT`: number of requests per minute, default is `10`; config file will overwrite this;
- `TOKENS_PER_MIN_LIMIT`: number of tokens per minute, default is `40000`; config file will overwrite this;
- `COROTINE_PER_AUTH`: number of corotine per api_key, default is `3`; decrease it to 1 if ratelimit errors are triggered too often;
- `ATTEMPTS_PER_PROMPT`: number of attempts per prompt, default is `5`;
- `RATELIMIT_AFTER_SUBMISSION`: whether to track ratelimit after submission, default is `True`; keep it enabled if response takes a long time;
- `OPENAI_LOG_LEVEL`: default log level is WARNING, 10-DEBUG, 20-INFO, 30-WARNING, 40-ERROR, 50-CRITICAL; set to 10 if you are getting stuck and want to do some diagnose;

Rate limit triggers will be visible under `logging.WARNING`. Run `export OPENAI_LOG_LEVEL=40` to ignore rate limit warnings if you believe current setting is stable enough.

### Performance Assessment

After ChatCompletion release, the `code-davinci-002` endpoint becomes slow. Using 10 API keys, running 100 completions with `max_tokens=20` and other hyperparameters left default took 90 seconds on average. Using official API, it took 10 seconds per completion, thus 1000 in total. 

Theroticallly, the throughput **increases linearly** with the number of API keys. 

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
   
2. Q: Why don't we just use server-less service (e.g. [Cloudflare Workers](https://workers.cloudflare.com/), [Tencent Cloud Functions](https://www.tencentcloud.com/products/scf)) to do the same thing?

   A: First, I usually write in Python, and most cloud services do not support Python server-less function. Second, I am not sure server-less solutions are capable of handling **rate limit controls** given their status-less nature. Tracking usage of each API key would be difficult (practical but not elegant) for server-less solutions.

### Acknowledgement

- [openai-cookbook](https://github.com/openai/openai-cookbook): Best practice when dealing with official APIs.
- [openai-python](https://github.com/openai/openai-python): Official Python version of OpenAI.

### TODO

#### Features

- [ ] Support all functions in OpenAI Python API.
  - [x] Completions
  - [x] Embeddings
  - [ ] Generations
  - [x] ChatCompletions
- [x] Better back-off strategy for maximum throughput.
- [x] Properly handling exceptions raised by OpenAI API.
- [x] Serving as a reverse proxy to balance official requests.
- [ ] Proxy-only mode to bypass blocks in Mainland China.

#### Advance Functions
- [ ] Automatic batching prompts to reduce the number of requests.
  - Now a request only carries a single prompt. We can implement **a splitting strategy to let it carry multiple prompts** while keeping token consumption not exceeding quotas.
- [ ] Automatic rotation of tons of OpenAI API Keys. (removing invaild, adding new, etc.)
- [ ] Distributed serving mode for concurrent requests via `python -m openai_manager.serving`.


### Donation

If this package helps your research, consider making a donation via GitHub! 
