import os
from functools import cache
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents import set_tracing_disabled
from agents import ModelSettings


@cache
def get_openai_model() -> OpenAIChatCompletionsModel:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = get_openai_client()
    return OpenAIChatCompletionsModel(model_name, openai_client=client)


@cache
def get_openai_client() -> AsyncOpenAI:
    chatai_api_key = os.getenv("CHATAI_API_KEY")
    openai_proxy_base_url = os.getenv("OPENAI_PROXY_BASE_URL")
    if chatai_api_key:
        set_tracing_disabled(True)
        return AsyncOpenAI(base_url=openai_proxy_base_url, api_key=chatai_api_key)
    else:
        return AsyncOpenAI()


@cache
def get_openai_model_settings():
    temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    return ModelSettings(temperature=temperature)
