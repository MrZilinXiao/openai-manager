import openai
import asyncio


async def main():
    prompt = "Once upon a time, "
    prompts = [prompt] * 10
    task_lst = []
    for prompt in prompts:
        print(f'Adding prompt...')
        task = await openai.Completion.acreate(model='text-davinci-003', prompt=prompt, max_tokens=20)
        task_lst.append(task)
    return await asyncio.gather(*task_lst)


def run():
    loop = asyncio.get_event_loop()
    task_lst = loop.run_until_complete(main())
    print([task.result() for task in task_lst])


if __name__ == '__main__':
    run()
