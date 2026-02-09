import json
import os


def _load_dotenv_if_available():
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)


class LLMClient:
    def __init__(self, model="gpt-4o"):
        _load_dotenv_if_available()
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "OpenAI SDK not available. Install openai and ensure OPENAI_API_KEY is set."
            ) from exc
        self.client = OpenAI()
        self.model = model

    def call_function(self, system_prompt, user_prompt, function_name, function_schema):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": function_schema.get("description", ""),
                    "parameters": function_schema["parameters"],
                },
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": function_name}},
            temperature=0,
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("No tool call returned by the model.")
        args_str = tool_calls[0].function.arguments
        return json.loads(args_str)
