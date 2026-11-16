import json

import httpx

OPENROUTER_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"

USER_PROMPT = """
Run a deep research to fact check this text. Identify any misleading information, questionable statements, or missing important information that would confuse the reader.

CRITICAL INSTRUCTIONS:
1. Respond in English
2. The "excerpt" field must quote ONE complete sentence, heading or list item from the original, copied EXACTLY as written: the same words, spelling and punctuation. Do not shorten it, do not use "...", do not join separate passages, and do not add brackets, explanations or any words of your own. Leave out a "(https://...)" link that follows the text.
3. Every issue MUST have a non-empty excerpt. If information is MISSING, quote the sentence or heading where it should be added and describe what is missing in the "issue" field.
4. Only include issues where you can point to specific problematic text OR identify specific gaps.
5. SOURCES FORMAT: Each entry in "sources" and "all_sources" MUST be a plain URL string (e.g. "https://example.com/path"). DO NOT use markdown such as [text](url), [text][url], or wrap the URL in parentheses or brackets. Output the bare URL only, with no surrounding characters.
6. Links in the text are written as "anchor text (https://...)". A statement followed by such a link cites that source: do not report it as having no source (you may still check whether the source supports it). Images appear as "[Image: description]"; you cannot see them, so do not report data as missing when it may be shown in an image.

Return your analysis as a JSON object with the following structure:
{
"issues": [
    {
    "excerpt": "ONE complete sentence copied exactly from the original",
    "issue": "explanation of what is wrong, misleading, or missing",
    "type": "misleading" | "questionable" | "incomplete",
    "sources": ["https://example.com/path-1", "https://example.com/path-2"]
    }
],
"all_sources": ["https://example.com/path-1", "https://example.com/path-2"]
}

For each issue, provide the bare URLs you used to verify the information. Include ALL bare URLs at the end in the all_sources array.

If no issues are found, return: {"issues": [], "all_sources": []}

Text to fact-check:
"""


def build_fact_check_body(model, text, now):
    """Body of the OpenRouter Responses API request for one fact check."""
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": f"You are a professional fact-checker. Today's date is {now.date().isoformat()}. Your training data is older than that, so verify every time-sensitive claim with web search before judging it. Analyze texts thoroughly and identify misleading information, questionable statements, and missing context. Always respond in English.",
            },
            {"role": "user", "content": f"{USER_PROMPT}{text}\n"},
        ],
        "stream": True,
        "usage": {"include": True},
        "reasoning": {"effort": "high"},
        # A server tool, not the `web` plugin: the key's zero-data-retention guardrail keeps
        # requests off OpenAI's own endpoint, so the plugin's native search never runs and it
        # falls back to a single 5-result search. The tool lets the model search per claim.
        #
        # No search limit: max_total_results defaults to 50 (empty searches after the tenth) and
        # the default 30-step budget is replaced by stop_server_tools_when, kept only as a runaway
        # guard. Exa charges the same per search up to 10 results, and 10 needed fewer searches.
        "tools": [
            {
                "type": "openrouter:web_search",
                "parameters": {"engine": "exa", "max_results": 10, "max_total_results": 1000},
            }
        ],
        "stop_server_tools_when": [{"type": "max_cost", "max_cost_in_dollars": 5}],
    }


def stream_fact_check(api_key, body, on_search):
    """Send the request and consume its SSE stream.

    Calls on_search({"query", "sources"}) for every completed web search. Returns
    {"text", "usage", "searches", "reasoning"}; usage.cost is what OpenRouter
    charged, web search fees included.
    """
    searches = []
    reasoning = ""
    text = ""
    timeout = httpx.Timeout(30.0, read=600.0)

    with httpx.stream(
        "POST",
        OPENROUTER_RESPONSES_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout,
    ) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(f"API error: {response.status_code} - {response.text}")

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            event_type = event.get("type")

            if event_type in ("response.reasoning_summary_text.delta", "response.reasoning.delta"):
                reasoning += event.get("delta", "")
            elif event_type == "response.output_text.delta":
                text += event.get("delta", "")
            elif event_type == "response.output_item.done":
                item = event.get("item", {})
                if item.get("type") == "openrouter:web_search" and item.get("action"):
                    action = item["action"]
                    search = {
                        "query": action.get("query") or "Web search",
                        "sources": [
                            {"url": source["url"], "title": source.get("title") or source["url"]}
                            for source in action.get("sources", [])
                        ],
                    }
                    searches.append(search)
                    on_search(search)
            elif event_type in ("response.failed", "response.incomplete"):
                failed = event.get("response", {})
                raise RuntimeError(
                    (failed.get("status_details") or {}).get("error", {}).get("message")
                    or (failed.get("incomplete_details") or {}).get("reason")
                    or (event.get("error") or {}).get("message")
                    or "AI model returned an error"
                )
            elif event_type in ("response.completed", "response.done"):
                completed = event.get("response", {})
                for item in completed.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text" and content.get("text"):
                                text = content["text"]
                return {"text": text, "usage": completed.get("usage", {}), "searches": searches, "reasoning": reasoning}

    raise RuntimeError("AI model stream ended unexpectedly. Please try again.")
