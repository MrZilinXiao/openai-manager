## OpenAI-Manager

Speed up your OpenAI requests by balancing prompts to multiple API keys. Quite useful if you are playing with `code-davinci-002` endpoint.

### Disclaimer

Before using this tool, you are required to read the EULA and ToS of OpenAI L.P. carefully. Actions that violate the OpenAI user agreement may result in the API Key and associated account being suspended. The author shall not be held liable for any consequential damages.

### Design


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

   2. YAML config file: you can add more fine-grained restrictions on each API key if you know the ratelimit for each key in advance.

3. Run this minimal running example to see how to boost your OpenAI completions. (more interfaces coming!)

```diff
- import openai
+ import openai_manager as openai

def test_batch_completion():
    prompt = "Once upon a time, "
    prompts = [prompt] * 10
    # openai_manager provides identical call signitures with official OpenAI Python API
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompts,
        max_tokens=20,
    )
    assert len(response["choices"]) == 10
    for i, answer in enumerate(response["choices"]):
        print("Answer {}: {}".format(i, answer["text"]))

if __name__ == '__main__':
    test_batch_completion()
```


### Performance Assessment

WIP

### Acknowledgement


### TODO

- [ ] Support all functions in OpenAI Python API.
  - [ ] Completions
  - [ ] Embeddings
  - [ ] Generations
- [ ] Better back-off strategy for maximum throughput.
- [ ] Properly handling exceptions raised by OpenAI API.


### Donation

If this package helps your research, consider making a donation via GitHub! 