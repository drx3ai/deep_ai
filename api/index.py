import os
import json
from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.tools import get_current_weather

load_dotenv(".env.local")

app = FastAPI()

# تعريف دالة لإنشاء عميل النموذج المناسب
def get_model_client(model_id: str):
    if model_id == 'deepseek':
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
    elif model_id == 'huggingface':
        return OpenAI(
            api_key=os.environ.get("HF_API_KEY"),
            base_url="https://api-inference.huggingface.co/models"
        )
    elif model_id == 'groq':
        return OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        # النموذج الافتراضي (DeepSeek)
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

# تعريف دالة للحصول على اسم النموذج المناسب
def get_model_name(model_id: str):
    if model_id == 'deepseek':
        return "deepseek-chat"
    elif model_id == 'huggingface':
        return "mistralai/Mistral-7B-Instruct-v0.2"  # مثال لنموذج Hugging Face
    elif model_id == 'groq':
        return "mixtral-8x7b-32768"  # مثال لنموذج Groq
    else:
        return "deepseek-chat"  # الافتراضي

class Request(BaseModel):
    messages: List[ClientMessage]
    model_id: str = 'deepseek'  # إضافة حقل model_id

available_tools = {
    "get_current_weather": get_current_weather,
}

def stream_text(messages: List[ChatCompletionMessageParam], model_id: str, protocol: str = 'data'):
    draft_tool_calls = []
    draft_tool_calls_index = -1
    
    # الحصول على العميل واسم النموذج المناسب
    client = get_model_client(model_id)
    model_name = get_model_name(model_id)
    
    # إنشاء الدفق مع النموذج المحدد
    stream = client.chat.completions.create(
        messages=messages,
        model=model_name,
        stream=True,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather at a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "The latitude of the location",
                        },
                        "longitude": {
                            "type": "number",
                            "description": "The longitude of the location",
                        },
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }] if model_id == 'deepseek' else None  # الأدوات متاحة فقط لـ DeepSeek حالياً
    )

    for chunk in stream:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue

            elif choice.finish_reason == "tool_calls":
                for tool_call in draft_tool_calls:
                    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"])

                for tool_call in draft_tool_calls:
                    tool_result = available_tools[tool_call["name"]](
                        **json.loads(tool_call["arguments"]))

                    yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"],
                        result=json.dumps(tool_result))

            elif choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    if (id is not None):
                        draft_tool_calls_index += 1
                        draft_tool_calls.append(
                            {"id": id, "name": name, "arguments": ""})

                    else:
                        draft_tool_calls[draft_tool_calls_index]["arguments"] += arguments

            else:
                yield '0:{text}\n'.format(text=json.dumps(choice.delta.content))

        if chunk.choices == []:
            usage = chunk.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            yield 'e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
                reason="tool-calls" if len(
                    draft_tool_calls) > 0 else "stop",
                prompt=prompt_tokens,
                completion=completion_tokens
            )

@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)

    response = StreamingResponse(stream_text(openai_messages, request.model_id, protocol))
    response.headers['x-vercel-ai-data-stream'] = 'v1'
    return response
