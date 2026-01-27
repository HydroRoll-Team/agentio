import asyncio
import httpx

from prompt_toolkit import prompt

class LLM(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_base = "http://localhost:11434/"

    async def invoke(self, _prompt: str) -> str:
        invoke_endpoint = f"{self.api_base.strip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": _prompt,
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(invoke_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result["response"]

async def main(model_name: str):
    llm = LLM(model_name)
    loop = asyncio.get_event_loop()
    while True:
        user_input = await loop.run_in_executor(None, lambda: prompt("User: "))
        if user_input.lower() in {"exit", "quit"}:
            break
        response = await llm.invoke(user_input)
        print(f"LLM: {response}")
        

if __name__ == "__main__":
    model_name = "gemma3:4b"
    asyncio.run(main(model_name))
