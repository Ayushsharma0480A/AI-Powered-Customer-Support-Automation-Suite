🚀 Hive Assignment

Author: Ayush Sharma

📦 Components

Part A — Email Tagging Mini-System

Part B — Sentiment Analysis Prompt Evaluation

Part C — Mini-RAG for Knowledge Base Answering

🧩 Part A — Email Tagging Mini-System
1️⃣ Approach

LLM-based classification using GPT-4o-mini

Inputs combined: subject + body

Classifier restricted to customer-specific tags only

Output format:

{ "tag": "...", "reason": "..." }

2️⃣ Ensuring Customer Isolation

A dictionary is created:

customer_id → allowed_tags


For each email:

Only tags belonging to that customer are allowed

Prevents cross-customer tag leakage

LLM prompt strictly enforces allowed_tags

3️⃣ Prompt Used
You are an email classifier. Choose exactly one tag from the allowed list.

Allowed tags: {tag_list}

Email Subject:
{subject}

Email Body:
{body}

Return ONLY valid JSON:
{
  "tag": "<one_of_allowed_tags>",
  "reason": "<short_reason>"
}

4️⃣ Error Analysis

Common issues identified:

Issue	Cause	Fix
Generic words misleading the model	“payment”, “urgent” appearing in many emails	Add phrase-level patterns
Politeness noise	“thanks”, “please”	Instruct model to ignore courtesy phrases
Ambiguous complaints	“email not working” matches multiple tags	Add context-heavy rules
5️⃣ Improvements (Production)

Add lightweight rule-based filters

Use customer-specific embeddings

Add human feedback loop for progressively improving accuracy

💬 Part B — Sentiment Analysis Prompt Evaluation
1️⃣ Prompt V1 (Initial)

Correctly classified all 10 emails

❌ 40% responses wrapped inside ```json markdown blocks

❌ Confidence inconsistent

2️⃣ Summary of V1 Results

Accuracy: 100%

Formatting problems: markdown wrapping

Neutral vs negative: sometimes inconsistent

Confidence: varied too widely

3️⃣ Improved Prompt V2 (Final)
You are a strict sentiment classifier.

Rules:
- Never return markdown or codeblocks.
- Return ONLY valid JSON.
- Frustration, urgency, anger → negative
- Appreciation, satisfaction → positive
- Simple questions without emotion → neutral
- Mixed emotions → neutral

Return:
{
  "sentiment": "...",
  "confidence": ...,
  "internal_reasoning": "..."
}

Email:
{email}

4️⃣ What Failed in V1

Codeblock-wrapped JSON

Loose emotional interpretation

Non-standard confidence ranges

5️⃣ What Improved in V2

Forced JSON-only output

Deterministic confidence ranges

Clear emotional classification rules

Higher consistency

6️⃣ How to Evaluate Prompts Systematically

Test with 10–50 diverse emails

Validate JSON structure

Compare prompts (A/B test)

Include edge cases (polite but angry, mixed sentiment)

Check stability under paraphrasing

📚 Part C — Mini-RAG for Knowledge Base Answering
1️⃣ Approach

Load KB files (.txt)

Extract embeddings using all-mpnet-base-v2

Build a FAISS L2 index

Retrieve top-k documents

Pass retrieved snippets to LLM for answer

Confidence = 1 - normalized_distance

2️⃣ Query Results
🔍 Query 1: “How do I configure automations in Hiver?”

Top Retrieved:

article1.txt

article5.txt

article4.txt

Generated Answer:

Go to Admin → Automations → Create Rule → Add Conditions → Add Actions → Enable.

Confidence: ~0.85

🔍 Query 2: “Why is CSAT not appearing?”

Top Retrieved:

article2.txt

article5.txt

article3.txt

Answer:

CSAT may be disabled or waiting to sync. Enable it under Admin → CSAT.

Confidence: ~0.78

3️⃣ Five Improvements

Chunk long documents (256–512 tokens)

Hybrid search (BM25 + embeddings)

Cross-encoder re-ranking

Metadata filtering (automation, analytics, auth)

Train domain-specific embeddings

4️⃣ Failure Case

Query: “Why are my emails delayed?”
Issue: KB did not contain “delay” topic → retrieval irrelevant.
Fix:

Add missing KB topics

Expand synonyms (lag, slow, stuck)

Hybrid lexical + semantic search

▶️ How to Run
pip install -r requirements.txt

# Part A
python src/classifier.py

# Part B
python src/sentiment_prompt_test.py

# Part C
python src/rag.py

📂 Folder Structure
hiver-assignment/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── classifier.py
│   ├── sentiment_prompt_test.py
│   └── rag.py
│
├── data/
│   ├── emails.csv
│   ├── sentiment_test_emails.txt
│   └── kb_articles/
│       ├── article1.txt
│       ├── article2.txt
│       ├── article3.txt
│       ├── article4.txt
│       ├── article5.txt