# src/classifier.py
import os
import pandas as pd
import json
import time
from openai import OpenAI

# Load API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMAILS_CSV = "data/emails.csv"


def build_prompt(subject, body, allowed_tags):
    tag_list = ", ".join(allowed_tags)

    prompt = f"""
You are an email classifier. Choose exactly one tag from the allowed list.

Allowed tags: {tag_list}

Email Subject:
{subject}

Email Body:
{body}

Return output as valid JSON ONLY:
{{
  "tag": "<one of allowed_tags>",
  "reason": "<short reasoning>"
}}
"""
    return prompt


def classify_with_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",    # change this if needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )

    return response.choices[0].message.content


def run_baseline():
    df = pd.read_csv(EMAILS_CSV, dtype=str).fillna("")

    # Build allowed tag list per customer
    tags_by_customer = {}
    for _, row in df.iterrows():
        cid = row["customer_id"]
        tags_by_customer.setdefault(cid, set()).add(row["tag"])

    results = []

    for _, row in df.iterrows():
        cid = row["customer_id"]
        allowed = sorted(list(tags_by_customer[cid]))

        prompt = build_prompt(row["subject"], row["body"], allowed)

        print("\n=== Running LLM for customer", cid, "===")

        output = classify_with_llm(prompt)

        results.append({
            "customer_id": cid,
            "subject": row["subject"],
            "body": row["body"],
            "ground_truth": row["tag"],
            "prediction": output
        })

        time.sleep(0.4)  # avoid rate limits

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_baseline()
