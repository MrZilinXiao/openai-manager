## OpenAI-Manager

Speed up your OpenAI requests by balancing prompts to multiple API keys. Quite useful if you are playing with `code-davinci-002` endpoint.

**If you seldomly trigger rate limit errors, it is unnecessary to use this package.**

### Disclaimer

Before using this tool, you are required to read the EULA and ToS of OpenAI L.P. carefully. Actions that violate the OpenAI user agreement may result in the API Key and associated account being suspended. The author shall not be held liable for any consequential damages.

### Design
TL;DR: this package helps you manage rate limit (both request-level and token-level) for each api_key for maximum number of requests to OpenAI API.

This is extremely helpful if you use `CODEX` endpoint or you have a handful of free-trial accounts due to limited budget. Free-trial accounts apply strict rate limit.

### Quickstart

1. Install openai-manager on PyPI.
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

   2. YAML config file: you can add more fine-grained restrictions on each API key if you know the ratelimit for each key in advance. (WIP)

3. Run this minimal running example to see how to boost your OpenAI completions. (more interfaces coming!)

    ```python
    import openai as official_openai
    import openai_manager
    
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


### Performance Assessment

WIP


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
   
   A: `code-davinci-002` or other similar OpenAI endpoints apply strict token-level rate limit, even if you upgrade to pay-as-you-go user. Simple batching would not solve this.

### Acknowledgement

[openai-cookbook](https://github.com/openai/openai-cookbook)

[openai-python](https://github.com/openai/openai-python)

### TODO

- [ ] Support all functions in OpenAI Python API.
  - [x] Completions
  - [ ] Embeddings
  - [ ] Generations
  - [ ] ChatCompletions
- [ ] Better back-off strategy for maximum throughput.
- [ ] Properly handling exceptions raised by OpenAI API.
- [ ] Automatic rotation of tons of OpenAI API Keys. (Removing invaild, adding new, etc.)


### Donation

If this package helps your research, consider making a donation via GitHub! 