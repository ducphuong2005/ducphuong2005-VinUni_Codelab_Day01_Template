"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
     
"""

import os
import time
from typing import Any, Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    # Khởi tạo client — đổi base_url sang ShopAIKey nếu không dùng OpenAI trực tiếp
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1"),
    )

    # Bấm giờ bắt đầu để đo latency
    start_time = time.time()

    # Gửi request đến Chat Completions API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Tính thời gian phản hồi
    latency = time.time() - start_time

    # Trích xuất text và token usage
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# NOTE: Lab variant — dùng OpenAI API (gpt-4o-mini) thay cho Gemini SDK
#       để đảm bảo chạy được với 1 API key duy nhất.
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    # Khởi tạo Gemini client bằng Google GenAI SDK
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Cấu hình tham số sinh văn bản
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    # Bấm giờ bắt đầu để đo latency
    start_time = time.time()

    # Gọi API sinh nội dung
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    # Tính thời gian phản hồi
    latency = time.time() - start_time

    # Trích xuất text và token usage từ metadata
    text = response.text or ""
    usage_meta = getattr(response, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
        "output_tokens": getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# NOTE: Lab variant — dùng OpenAI API (gpt-4o-mini) thay cho Anthropic SDK
#       để đảm bảo chạy được với 1 API key duy nhất.
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic

    # Khởi tạo Anthropic client với API key từ biến môi trường
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Bấm giờ bắt đầu để đo latency
    start_time = time.time()

    # Gọi Messages API của Anthropic
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )

    # Tính thời gian phản hồi
    latency = time.time() - start_time

    # Trích xuất text và token usage
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str = "") -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # 1. Gọi GPT-4o
    resp_4o, lat_4o, usage_4o = call_openai(prompt, model=OPENAI_MODEL)
    in_4o = usage_4o.get("input_tokens", 0)
    out_4o = usage_4o.get("output_tokens", 0)
    cost_4o = (
        in_4o * PRICING_1M_TOKENS["gpt-4o"]["input"]
        + out_4o * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    # 2. Gọi GPT-4o-mini
    resp_mini, lat_mini, usage_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    in_mini = usage_mini.get("input_tokens", 0)
    out_mini = usage_mini.get("output_tokens", 0)
    cost_mini = (
        in_mini * PRICING_1M_TOKENS["gpt-4o-mini"]["input"]
        + out_mini * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    # 3. Gọi Gemini 2.5 Flash
    resp_gem, lat_gem, usage_gem = call_gemini(prompt, model=GEMINI_MODEL)
    in_gem = usage_gem.get("input_tokens", 0)
    out_gem = usage_gem.get("output_tokens", 0)
    cost_gem = (
        in_gem * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"]
        + out_gem * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": resp_4o,
            "latency": lat_4o,
            "cost": cost_4o,
            "input_tokens": in_4o,
            "output_tokens": out_4o,
        },
        "gpt4o_mini": {
            "response": resp_mini,
            "latency": lat_mini,
            "cost": cost_mini,
            "input_tokens": in_mini,
            "output_tokens": out_mini,
        },
        "gemini_flash": {
            "response": resp_gem,
            "latency": lat_gem,
            "cost": cost_gem,
            "input_tokens": in_gem,
            "output_tokens": out_gem,
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with OpenAI GPT-4o
# NOTE: Lab variant — dùng OpenAI Streaming thay cho Gemini.
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1"),
    )

    history: list[dict] = []

    print("=== GPT-4o Streaming Chatbot Sẵn Sàng (gõ 'quit' hoặc 'exit' để thoát) ===")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Thêm tin nhắn của người dùng vào lịch sử
        history.append({"role": "user", "content": user_input})

        # Giữ tối đa 3 lượt gần nhất (6 tin nhắn: 3 user + 3 assistant)
        history = history[-6:]

        print("GPT-4o: ", end="", flush=True)
        full_response = ""

        try:
            # Gọi API ở chế độ streaming
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=history,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    print(delta, end="", flush=True)
                    full_response += delta
            print()
        except Exception as e:
            print(f"\n[Lỗi streaming: {e}]")
            # Xoá tin nhắn vừa thêm vào nếu gặp lỗi
            history.pop()
            continue

        # Lưu câu trả lời của model vào lịch sử
        history.append({"role": "assistant", "content": full_response})

        # Giới hạn lịch sử ở 3 lượt gần nhất
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Chờ exponential backoff trước khi thử lại
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        try:
            res = compare_models(prompt)
        except TypeError:
            res = compare_models()
        item = dict(res)
        item["prompt"] = prompt
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    model_mapping = [
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-Mini"),
        ("gemini_flash", "Gemini-Flash"),
    ]

    for item in results:
        prompt = item.get("prompt", "")
        for key, display_name in model_mapping:
            if key in item:
                stats = item[key]
                resp = str(stats.get("response", "")).replace("\n", " ").strip()
                if len(resp) > 50:
                    resp = resp[:47] + "..."
                latency = f"{stats.get('latency', 0.0):.2f}s"
                tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
                cost = f"${stats.get('cost', 0.0):.6f}"
                lines.append(f"| {prompt} | {display_name} | {resp} | {latency} | {tokens} | {cost} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting GPT-4o Streaming Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
