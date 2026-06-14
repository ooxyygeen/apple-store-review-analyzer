import json
import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")


def generate_insights(keywords: list[str]) -> dict[str, str]:
    """
    Generate actionable insights from negative review keywords using an LLM.

    Groups keywords into distinct topics and returns one concrete,
    actionable recommendation per topic as a dict where keys are
    snake_case topic names and values are insight strings.

    Uses LiteLLM for provider-agnostic LLM access — switch providers
    by changing LLM_MODEL and the corresponding API key in .env.

    Returns an empty dict if no keywords are provided.
    """
    if not keywords:
        return {}

    prompt = f"""You are a product analyst reviewing negative App Store user feedback.

Based on these keywords extracted from negative reviews (ordered by frequency):
{", ".join(keywords)}

Group the keywords into distinct topics and provide exactly one concise actionable insight per topic.

Each insight must:
- Identify the specific problem
- Suggest a concrete action for the development team
- Be 1-2 sentences max

Respond with a JSON object where each key is a short snake_case topic name and the value is the insight string.
Example format:
{{
    "billing_transparency": "Users are frequently surprised by charges after the free trial ends. Consider adding a clear reminder 24 hours before the trial expires.",
    "cancellation_flow": "Multiple users report difficulty cancelling their subscription. Add a visible cancellation option directly within the app settings."
}}

Return only the JSON object, no other text."""

    response = completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    if not isinstance(parsed, dict):
        raise ValueError(f"Unexpected response format from LLM: {content}")

    return parsed