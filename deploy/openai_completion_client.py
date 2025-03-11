# SPDX-License-Identifier: Apache-2.0
# reference: 
# - https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_completion_client.py
# - https://platform.openai.com/docs/api-reference/completions/object
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Completion API
stream = True
completion = client.completions.create(
    model=model,
    prompt="请你给我讲一个笑话:",
    echo=False,
    n=2,
    stream=stream,
    logprobs=3)

print("Completion results:")
if stream:
    for c in completion:
        print(c.choices[0].text, end="", flush=True)
    print()
else:
    print(completion)