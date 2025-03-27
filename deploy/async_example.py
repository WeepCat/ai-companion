import asyncio
from openai import AsyncOpenAI
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio
from openai import APIError, APITimeoutError, APIConnectionError, RateLimitError
import aiohttp
from dotenv import load_dotenv
load_dotenv("./evaluate/.env")
# 定义应该重试的网络相关错误
NETWORK_ERRORS = (
    APITimeoutError,    # 超时错误
    APIConnectionError, # 连接错误
    aiohttp.ClientError, # aiohttp 客户端错误
    RateLimitError,    # 速率限制错误
)

api_key = os.environ["OPENAI_API_KEY"],
base_url = os.environ['OPENAI_API_BASE_URL']
model = "gpt-4-turbo"
system_message = "You are a helpful assistant who is helping a user with a problem"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(NETWORK_ERRORS)  # 只重试网络相关错误
)
async def chat(history, client: AsyncOpenAI, semaphore):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": history}
                ]
            )
            return response.choices[0].message.content
        except NETWORK_ERRORS as e:
            raise e
        except APIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Unexpected Error: {str(e)}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(NETWORK_ERRORS)  # 只重试网络相关错误
)
async def completion(prompt, client: AsyncOpenAI, semaphore):
    async with semaphore:
        try:
            response = await client.completions.create(
                model=model,
                prompt=prompt,
                echo=False
            )
            return response.choices[0].text.strip()
        except NETWORK_ERRORS as e:
            raise e
        except APIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Unexpected Error: {str(e)}"
        

# 模拟异步地向OpenAI API发送请求, 最大并发数为10
async def main():
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(10)
    # demo_queries = ["What is the capital of France?", "What is the capital of Spain?", "What is the capital of Italy?"]
    demo_queries = ["What is the capital of France?", "What is the capital of Spain?", "What is the capital of Italy?"]
    results = []
    for history in tqdm_asyncio(demo_queries):
        result = await completion(history, client, semaphore)
        results.append(result)
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
