import logging
import os
from functools import cache

from agents import ModelSettings
from agents import OpenAIChatCompletionsModel
from agents import set_tracing_disabled
from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI


@cache
def get_openai_model() -> OpenAIChatCompletionsModel:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = get_openai_client()
    return OpenAIChatCompletionsModel(model_name, openai_client=client)


@cache
def get_openai_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    chatai_api_key = os.getenv("CHATAI_API_KEY")
    openai_proxy_base_url = os.getenv("OPENAI_PROXY_BASE_URL")
    if chatai_api_key:
        logging.info("Using ChatAI API key")
        set_tracing_disabled(True)
        return AsyncOpenAI(base_url=openai_proxy_base_url, api_key=chatai_api_key)
    elif os.getenv("AZURE_OPENAI_API_KEY"):
        logging.info("Using Azure OpenAI API key")
        set_tracing_disabled(True)
        return AsyncAzureOpenAI(api_version=os.getenv("OPENAI_API_VERSION", "2023-05-15"))
    else:
        return AsyncOpenAI()


@cache
def get_openai_model_settings():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = None if model == "o3-mini" else float(os.getenv("OPENAI_TEMPERATURE", 0.0))
    return ModelSettings(temperature=temperature)
