import openai_manager as openai
import os
# import parallel openai placeholder

api_key = os.environ.get("OPENAI_API_KEY")

def test_batch_completion():
    prompt = "Once upon a time, "
    prompts = [prompt] * 10
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompts,
        max_tokens=20,
    )
    assert len(response["choices"]) == 10
    for i, answer in enumerate(response["choices"]):
        print("Answer {}: {}".format(i, answer["text"]))