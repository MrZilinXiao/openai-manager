# build 100 requests and see benchmark results (openai v.s. openai-manager)
import openai as official_openai
import openai_manager
from openai_manager.utils import timeit


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
def test_manager(count=10):
    prompt = "Once upon a time, "
    prompts = [prompt] * count
    responses = openai_manager.Completion.create(
        model="code-davinci-002",
        prompt=prompts,
        max_tokens=20,
    )
    assert len(
        responses) == count, f"Length do not match: {len(responses)} vs {count}"
    for i, response in enumerate(responses):
        print("Answer {}: {}".format(i, response["choices"][0]["text"]))


if __name__ == '__main__':
    # allow tests without pytest
    # print('--------Official---------')
    # test_official_separate()
    print('--------Manager---------')
    test_manager(count=100)
