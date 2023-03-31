import unittest
import openai
# import multiprocessing
# import os

# for multithread uvicorn solution
import uvicorn
import contextlib
import time
import threading


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

# def create_serving_process():
#     """A ugly os patch to run command in a separate process"""
#     def _run():
#         os.system("python -m openai_manager.serving --port 8000 --host localhost --api_key sk-test_openai_manager_serving")
#     p = multiprocessing.Process(target=_run)
#     p.start()
#     return p


class TestServing(unittest.TestCase):
    def setUp(self) -> None:
        # self.serving_process = create_serving_process()
        from openai_manager.serving import app
        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "sk-test_openai_manager_serving"
        app.api_key = "sk-test_openai_manager_serving"
        config = uvicorn.Config(
            app=app, host="localhost", port=8000, log_level="info")
        self.server = Server(config=config)

    def test_openai_create(self):
        with self.server.run_in_thread():
            prompt = ["Once upon a time, "] * 5
            response = openai.Completion.create(
                model="text-ada-001",
                prompt=prompt,
                max_tokens=10,
            )
            print(response)
            self.assertEqual(len(response['choices']), 5)


if __name__ == '__main__':
    unittest.main()
