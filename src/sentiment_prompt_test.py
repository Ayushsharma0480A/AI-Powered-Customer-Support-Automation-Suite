import os
import json
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMAILS_FILE = "data/sentiment_test_emails.txt"

# ----------------------------
# Prompt Template (V1)
# ----------------------------
prompt_v1_template = """
You are a sentiment classifier for customer-support emails.

Return a JSON object with:
- sentiment: positive, neutral, or negative
- confidence: a float between 0 and 1
- internal_reasoning: short explanation (for debugging)

Email:
{email}

Return ONLY valid JSON.
"""


# ----------------------------
# FIXED email block splitter
# ----------------------------
def read_emails(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Split by two or more newlines — safest approach
    parts = re.split(r"\n\s*\n", content)

    # Clean blocks
    emails = [p.strip() for p in parts if p.strip()]
    return emails


# ----------------------------
# LLM Classifier
# ----------------------------
def classify(email):
    prompt = prompt_v1_template.format(email=email)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    txt = response.choices[0].message.content

    # Try to parse JSON output
    try:
        return json.loads(txt)
    except:
        return {"raw_output": txt}


# ----------------------------
# Runner
# ----------------------------
def run():
    emails = read_emails(EMAILS_FILE)
    results = []

    for i, email in enumerate(emails, start=1):
        print(f"\n=== Processing Email {i} ===")
        out = classify(email)
        results.append({"email": email, "result": out})

    print("\nFINAL RESULTS:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
