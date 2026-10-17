import json


def parse_json_object(text):
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        start = text.find("{") if isinstance(text, str) else -1
        if start < 0:
            raise ValueError("LLM 没有返回有效 JSON，请重新生成") from None
        try:
            result, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            raise ValueError("LLM 返回的 JSON 格式不完整，请重新生成") from None
    if not isinstance(result, dict):
        raise ValueError("LLM 返回结果应为 JSON 对象")
    return result
