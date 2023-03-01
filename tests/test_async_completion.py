import parallel_openai as openai
import os
import asyncio
# import parallel openai placeholder

api_key = os.environ.get("OPENAI_API_KEY")

async def self_async_completion():
    prompt = "Once upon a time, "
    prompts = [prompt] * 10
    task_lst = []
    for i, prompt in enumerate(prompts):
        task = await openai.Completion.async_create(
            model="code-davinci-002",
            prompt=prompt,
            max_tokens=20,
        )
        task_lst.append(task)
    return await asyncio.gather(*task_lst)

def test_async_completion():
    loop = asyncio.get_event_loop()
    task_lst = loop.run_until_complete(self_async_completion())
    print(task_lst)