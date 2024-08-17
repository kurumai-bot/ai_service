from enum import Enum
import logging
from typing import Any, Generator, Type

from guidance import assistant, capture, gen, guidance, json, select, system, user, with_temperature
from guidance.chat import (
    ChatTemplate,
    Llama2ChatTemplate,
    Llama3ChatTemplate,
    Mistral7BInstructChatTemplate,
    Phi3ChatTemplate
)
from guidance.models import LlamaCpp, Model, OpenAI, Transformers
from guidance.models._model import Engine
from guidance.models._openai import OpenAIEngine
from guidance.models.llama_cpp._llama_cpp import LlamaCppEngine
from guidance.models.transformers._transformers import TransformersEngine
import orjson
from pydantic import BaseModel


# T = TypeVar("T")


# class TextGenInference(Generic[T]):
#     def __init__(self, **kwargs) -> None:
#         pass

#     def generate_from_prompt(
#         self,
#         prompt: str,
#         context: T,
#         **kwargs
#     ) -> Generator[str, T, None] | Tuple[str, T]:
#         pass

#     def encode(self, string: str) -> List[int]:
#         pass

#     def decode(self, tokens: List[int]) -> str:
#         pass

#     def trim_context(self, context: T, token_limit: int = 1_000) -> T:
#         pass


class Emotion(str, Enum):
    happy = "happy",
    sad = "sad",
    angry = "angry",
    neutral = "neutral"


class Response(BaseModel):
    emotion: Emotion
    message: str
    should_ask_for_user_response: bool


# TODO: Experiment with per word or timed emotions
@guidance(stateless=True)
def json_gen(lm, name=None, temperature: int = 1):
    lm += capture(f"""\
    {{
        "emotion": "{select(["happy", "sad", "neutral", "angry"])}",
        "message": "{gen(stop=['"'], temperature=temperature)}",
        "user_response_needed": {select(["true", "false"], name="user_response_needed")}
    }}""", name=name)
    return lm


class TextGenProcessor:
    def __init__(self, engine: str | Engine, **kwargs) -> None:
        self.logger = kwargs.pop("logger", None) or logging.getLogger("text_gen")

        self.max_context_tokens: int = kwargs.pop("max_context_tokens", 500)
        context = kwargs.pop("context", "")

        # Initialize inference
        self.inference: Engine = None
        self._model: Model = None
        self.inference: Any = None

        if isinstance(engine, str):
            # Get best chat template
            engine_lower = engine.lower()
            chat_template: Type[ChatTemplate] = None
            if "llama-2" in engine_lower:
                chat_template = Llama2ChatTemplate
            elif "llama-3" in engine_lower:
                chat_template = Llama3ChatTemplate
            elif "mistral-7b-instruct" in engine_lower:
                chat_template = Mistral7BInstructChatTemplate
            elif "phi-3" in engine_lower:
                chat_template = Phi3ChatTemplate

            # Create model and assign inference
            model_type, model_name = engine.split("/", 1)
            openai_api_key = kwargs.pop("openai_api_key", None)
            hf_api_key = kwargs.pop("hf_api_key", None)
            if model_type == "openai":
                self._model = OpenAI(model_name, api_key=openai_api_key, **kwargs)
            elif model_type == "transformers":
                self._model = Transformers(
                    model_name,
                    chat_template=chat_template,
                    token=hf_api_key,
                    **kwargs
                )
            elif model_type == "llamacpp":
                self._model = LlamaCpp(model_name, chat_template=chat_template, **kwargs)
            else:
                raise ValueError("Passed in model name was unrecognized: %s", engine)
            self.inference = self._model.engine
        else:
            if isinstance(engine, OpenAIEngine):
                self._model = OpenAI(
                    engine.model_name,
                    api_key=kwargs.pop("openai_api_key", None),
                    **kwargs
                )
            elif isinstance(engine, TransformersEngine):
                self._model = Transformers(
                    engine.model_obj,
                    tokenizer=engine.tokenizer._orig_tokenizer,
                    chat_template=type(engine.get_chat_template()),
                    **kwargs
                )
            elif isinstance(engine, LlamaCppEngine):
                self._model = LlamaCpp(
                    engine.model_obj,
                    chat_template=type(engine.get_chat_template()),
                    **kwargs
                )
            else:
                raise ValueError("Passed in model is unsupported.")
            self.inference = engine

        # Don't print model output into the console
        self._model.echo = False

        with system():
            self._model += context

        self.logger.info(
            "Text Gen Processor initialized with model `%s`.",
            self.inference.__class__.__name__
        )

    def get_state(self) -> str:
        return str(self._model)

    def generate_from_prompt(self, prompt: str, **kwargs) -> Generator[Any, None, None] | list[Any]:
        self.trim_context()
        with user():
            self._model += prompt

        temperature = kwargs.pop("temperature", 0.5)

        def generator():
            with assistant():
                user_response_needed = False
                while not user_response_needed:
                    
                    # Gen new text
                    self._model += json(name="response", schema=Response, temperature=temperature)
                    response = orjson.loads(self._model["response"])

                    # Clean response object
                    user_response_needed = response.pop("should_ask_for_user_response")

                    yield response

        if kwargs.pop("stream", False):
            return generator()
        else:
            return list(generator())

    def trim_context(self):
        # TODO: while this probably works for now, clean it up later
        self._model._state = self._model._state[:self.max_context_tokens * 3]

    def clear_context(self):
        self._model.reset()


# class OpenAI(TextGenInference[str]):
#     def __init__(self, model: str, **kwargs) -> None:
#         super().__init__(**kwargs)

#         self.model = model

#         self.client = Client(api_key=kwargs.pop("openai_api_key", None))

#         try:
#             self.tokenizer = tiktoken.encoding_for_model(model)
#         except KeyError:
#             # Most likely a fine tune, default to r50k_base
#             self.tokenizer = tiktoken.get_encoding("r50k_base")

#     def generate_from_prompt(
#         self,
#         prompt: str,
#         context: str,
#         **kwargs
#     ) -> Generator[str, str, None] | Tuple[str, str]:
#         # Add model, api key, and prompt into request
#         if "model" not in kwargs:
#             kwargs["model"] = self.model

#         prompt = context + prompt
#         kwargs["prompt"] = prompt

#         # Return a generator that just extracts the content of each event if streaming
#         if kwargs.get("stream", False):
#             def generator():
#                 # A list to store newly generated text
#                 chunks = [prompt]
#                 for chunk in self.client.completions.create(**kwargs):
#                     chunks.append(chunk.choices[0].text)
#                     yield chunk.choices[0].text

#                 # Return full new text
#                 return "".join(chunks)
#             return generator()

#         res = self.client.completions.create(**kwargs).choices[0].text
#         return (res, context + res)

#     def encode(self, string: str) -> List[int]:
#         return self.tokenizer.encode(string)

#     def decode(self, tokens: List[int]) -> str:
#         return self.tokenizer.decode(tokens)

#     def trim_context(self, context: str, token_limit: int = 1_000) -> str:
#         # Ensure context is a string
#         if context is None:
#             context = ""
#         elif not isinstance(context, str):
#             raise ValueError("OpenAI inference context must be a string.")

#         # Trim tokens by converting to tokens and just taking however many
#         # tokens from the end of the tokens and decode that into text
#         tokens = self.encode(context)
#         return self.decode(tokens[-token_limit:])


# class OpenAIChat(TextGenInference[list]):
#     def __init__(self, model: str, **kwargs) -> None:
#         super().__init__(**kwargs)

#         self.model = model

#         self.client = Client(api_key=kwargs.pop("openai_api_key", None))

#         self.tokenizer = tiktoken.encoding_for_model(model)

#     def generate_from_prompt(
#         self,
#         prompt: str,
#         context: list,
#         **kwargs
#     ) -> Generator[str, list, None] | Tuple[str, list]:
#         # Add model, api key, and messages into request
#         if "model" not in kwargs:
#             kwargs["model"] = self.model

#         # Don't want to modify in place, copy instead
#         messages = context.copy()
#         messages.append({"role": "user", "content": prompt})
#         kwargs["messages"] = messages

#         # Return a generator that just extracts the content of each event if streaming
#         if kwargs.get("stream", False):
#             def generator():
#                 chunks = []
#                 role = None

#                 # The first event will return the name of the role,
#                 # the subsequent events will be the actual text
#                 for chunk in self.client.chat.completions.create(**kwargs):
#                     delta = chunk.choices[0].delta
#                     if delta.role is not None:
#                         role = delta.role
#                     if delta.content is not None and delta.content != "":
#                         chunks.append(delta.content)
#                         yield delta.content

#                 messages.append({"role": role, "content": "".join(chunks)})
#                 # Return full new text
#                 return messages
#             return generator()

#         res = self.client.chat.completions.create(**kwargs).choices[0].message
#         messages.append(res)
#         return (res.content, messages)

#     def encode(self, string: str) -> List[int]:
#         return self.tokenizer.encode(string)

#     def decode(self, tokens: List[int]) -> str:
#         return self.tokenizer.decode(tokens)

#     def trim_context(self, context: list, token_limit: int = 1_000) -> list:
#         # Ensure context is a list
#         if context is None:
#             context = []
#         elif not isinstance(context, list):
#             raise ValueError("OpenAI chat inference context must be a list.")

#         # Trim tokens by counting tokens from newest to oldest, then
#         # removing any message that goes over. So it won't trim exactly to the limit,
#         # but close enough
#         # Based off https://platform.openai.com/docs/guides/gpt/managing-tokens
#         # TODO: Actually respect the limit rather than get "close enough" (ur dumb past me)
#         token_count = 0
#         res = []
#         for message in reversed(context):
#             token_count += 4
#             for key, value in message.items():
#                 if value is None or value == "":
#                     continue
#                 token_count += len(self.encode(value))
#                 if key == "name":
#                     token_count -= 1
#             token_count += 2

#             if token_count <= token_limit:
#                 res.insert(0, message)
#             else:
#                 break

#         return res


# class TransformersInference(TextGenInference[str]):
    # def __init__(self, model: str, **kwargs) -> None:
    #     super().__init__(**kwargs)

    #     self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    #     self.tokenizer = AutoTokenizer.from_pretrained(model)
    #     self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
    #     self.pipeline: TextGenerationPipeline = pipeline(
    #         task="text-generation", 
    #         model=self.model, 
    #         tokenizer=self.tokenizer,
    #         device=kwargs.get("device")
    #     )

    #     self.human_string = kwargs.pop("human_string", "")
    #     self.robot_string = kwargs.pop("robot_string", "")

    # def generate_from_prompt(
    #     self,
    #     prompt: str,
    #     context: T,
    #     **kwargs
    # ) -> Generator[str, T, None] | Tuple[str, T]:
    #     prompt = self.human_string + prompt + self.robot_string

    #     # Return the streamer and start the pipeline in a diff thread
    #     if kwargs.pop("stream", False):
    #         # While I don't like making a new thread, it seems this is necessary
    #         Thread(name="generation_thread", target=self.pipeline, args=(prompt,), kwargs={
    #             "prefix": context,
    #             "max_length": 1_000,
    #             "streamer": self.streamer,
    #             **kwargs
    #         }).start()
    #         def generator():
    #             # A list to store newly generated text
    #             chunks = [context, prompt]
    
    #             for word in self.streamer:
    #                 # TODO: This is a monkeypatch for llama2's eos(?) token. Fix this later
    #                 word = word.removesuffix("</s>")
    #                 chunks.append(word)
    #                 yield word

    #             # Return full new text
    #             return "".join(chunks)
    #         return generator()
    #     else:
    #         res = self.pipeline(
    #             prompt,
    #             return_full_text=False,
    #             prefix=context,
    #             max_length=1_000,
    #             **kwargs
    #         )
    #     text = res[0]["generated_text"]
    #     return (text, context + text)

    # def encode(self, string: str) -> List[int]:
    #     return self.tokenizer.encode(string)

    # def decode(self, tokens: List[int]) -> str:
    #     return self.tokenizer.decode(tokens)

    # def trim_context(self, context: str, token_limit: int = 1_000) -> str:
    #     # Ensure context is a string
    #     if context is None:
    #         context = ""
    #     elif not isinstance(context, str):
    #         raise ValueError("TransformersInference context must be a string.")

    #     # Trim tokens by converting to tokens and just taking however many
    #     # tokens from the end of the tokens and decode that into text
    #     tokens = self.encode(context)
    #     return self.decode(tokens[-token_limit:])


if __name__ == "__main__":
    processor = TextGenProcessor("georgesung/llama2_7b_chat_uncensored", human_string="\n\n### HUMAN:\n", robot_string="\n\n### RESPONSE:\n", max_context_tokens=1000, openai_api_key=OPENAI_API_KEY, device=0)
    processor.context = """Enter RP mode. Pretend to be a college frat boy.

You shall reply to the user while staying in character.
"""
    print(processor.context)
    while True:
        print("\n\n### HUMAN:\n", end="")
        thing = input()
        print("\n\n### RESPONSE:\n", end="")
        for word in processor.generate_from_prompt(thing, stream=True, repetition_penalty=1.2, length_penalty=0.8):
            print(word, end="")
    # for token in processor.generate_from_prompt(input("what") + ". respond with a series of messages separated by \"<eom>\"", stream=True):
    #     print(token, end="")


# pylint: disable-all
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, pipeline, StoppingCriteria


# class TextGenProcessor:
#     def __init__(self, model: str, device: str = "cuda") -> None:
#         self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model, low_cpu_mem_usage=True)
#         print(self.model.eval())

#         self.tokenizer = AutoTokenizer.from_pretrained(model)

#         # self.pipeline = pipeline(
#         #     task="text-generation",
#         #     model=self.model,
#         #     framework="pt",
#         #     device=torch.device(device),
#         #     stopping_criteria=_StoppingCriteria()
#         # )

#     def generate(self, prompt: str) -> str:
#         return ""


# class _StoppingCriteria(StoppingCriteria):
#     def __init__(self) -> None:
#         super().__init__()

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         print(input_ids)
#         return True


# if __name__ == "__main__":
#     processor = TextGenProcessor("Neko-Institute-of-Science/pygmalion-7b")
#     while True:
#         print(processor.generate(input("im so confused")))
