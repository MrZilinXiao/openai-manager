# build 100 requests and see benchmark results (openai v.s. openai-manager)
import openai as official_openai
import openai_manager
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@timeit
def test_official_batch():
    prompt = "Once upon a time, "
    prompts = [prompt] * 10
    response = official_openai.Completion.create(
        model="code-davinci-002",
        prompt=prompts,
        max_tokens=20,
    )
    assert len(response["choices"]) == 10
    for i, answer in enumerate(response["choices"]):
        print("Answer {}: {}".format(i, answer["text"]))


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


if __name__ == '__main__':
    # allow tests without pytest
    # print('--------Official---------')
    # test_official_separate()
    print('--------Manager---------')
    test_manager()
