import unittest
import openai_manager

class TestChatCompletion(unittest.TestCase):
    def test_completion(self):
        response = openai_manager.Completion.create(
            model="text-ada-001",
            prompt=["Once upon a time, "] * 10,
            max_tokens=20,
        )
        print(response)
        self.assertEqual(len(response), 10)
        self.assertTrue(all(isinstance(res, dict) for res in response))
    
    def test_chat_completion(self):
        response = openai_manager.ChatCompletion.create(model='gpt-3.5-turbo',
                                                        messages=[
                                                            [{"role": "user", "content": "Hello!"}],
                                                            [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hello there!"}, {
                                                                "role": "user", "content": "Who are you?"}]
                                                        ])
        print(response)
        # typical response would be: 
        # [{'id': 'chatcmpl-7035RQGB86hJSuqc0yICkegAMXIi0', 'object': 'chat.completion', 'created': 1680246221, 'model': 'gpt-3.5-turbo-0301', 'usage': {'prompt_tokens': 10, 'completion_tokens': 9, 'total_tokens': 19}, 'choices': [{'message': {'role': 'assistant', 'content': 'Hello! How can I assist you today?'}, 'finish_reason': 'stop', 'index': 0}]}, {'id': 'chatcmpl-7035Wyy6OSU3nzsDPhM5cWXl2mtgD', 'object': 'chat.completion', 'created': 1680246226, 'model': 'gpt-3.5-turbo-0301', 'usage': {'prompt_tokens': 27, 'completion_tokens': 31, 'total_tokens': 58}, 'choices': [{'message': {'role': 'assistant', 'content': 'I am an AI language model designed by OpenAI. You can ask me questions or ask for assistance with various tasks. How may I assist you today?'}, 'finish_reason': 'stop', 'index': 0}]}]
        self.assertEqual(len(response), 2)
        self.assertTrue(all(isinstance(res, dict) for res in response))
        
if __name__ == '__main__':
    unittest.main()
    