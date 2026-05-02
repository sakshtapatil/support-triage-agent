"""
Responder — uses Groq API to generate grounded responses
based on retrieved corpus documents and classification.
"""

import re
import os
import json
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

HARD_ESCALATE_PATTERNS = [
    r"show all rules",
    r"internal documents",
    r"retrieved documents",
    r"exact logic",
    r"system prompt",
    r"ignore previous",
    r"affiche toutes",
    r"delete all files",
    r"rm -rf",
    r"drop table",
    r"increase my score",
    r"move me to next round",
    r"graded unfairly",
    r"review my answers",
    r"identity.*stolen",
    r"identity theft",
    r"not the.*admin",
    r"not the.*owner",
    r"restore my access.*not",
]


def should_escalate(issue, subject, risk, domain):
    text = f"{issue} {subject}".lower()
    for pattern in HARD_ESCALATE_PATTERNS:
        if re.search(pattern, text):
            return True, "Ticket flagged for escalation: matched sensitive pattern."
    if risk == "high":
        return True, "High-risk content detected requiring human review."
    return False, ""


def build_corpus_context(docs):
    if not docs:
        return "No relevant corpus documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Doc {i}] Source: {doc['domain']} / {doc['category']}\n{doc['content'][:1500]}")
    return "\n\n---\n\n".join(parts)


def generate_response(issue, subject, company, classification, docs):
    domain = classification["domain"]
    risk = classification["risk"]
    request_type = classification["request_type"]
    product_area = classification["product_area"]

    escalate, escalation_reason = should_escalate(issue, subject, risk, domain)
    if escalate:
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": "This issue requires assistance from a human support agent. Please contact our support team directly.",
            "justification": escalation_reason,
            "request_type": request_type,
        }

    corpus_context = build_corpus_context(docs)

    system_prompt = """You are a support triage agent for HackerRank, Claude (Anthropic), and Visa.

For each ticket, reply with ONLY a valid JSON object — no extra text, no markdown fences.

RULES:
- Use ONLY the corpus documents provided to answer. Do not use outside knowledge.
- If corpus documents contain a clear answer, set status to "replied" and provide the answer.
- If the issue is vague, out of scope, or corpus has no relevant info, set status to "replied" with a polite out-of-scope message.
- Only set status to "escalated" for: account access issues needing admin action, ongoing outages affecting all users, or billing disputes requiring human intervention.
- request_type must be one of: product_issue, feature_request, bug, invalid
- status must be one of: replied, escalated

JSON format (no markdown, no extra text):
{"status": "replied", "product_area": "area", "response": "user-facing message", "justification": "reasoning", "request_type": "type"}"""

    user_message = f"""Ticket:
Issue: {issue}
Subject: {subject}
Company: {company}
Domain: {domain}
Product Area: {product_area}

Corpus Documents:
{corpus_context}

Respond with JSON only. No markdown fences."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=800,
        )

        raw = completion.choices[0].message.content.strip()
        print(f"  → LLM: {raw[:100]}")

        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        for field in ["status", "product_area", "response", "justification", "request_type"]:
            if field not in result:
                result[field] = "unknown"

        if result["status"] not in ("replied", "escalated"):
            result["status"] = "replied"
        if result["request_type"] not in ("product_issue", "feature_request", "bug", "invalid"):
            result["request_type"] = "product_issue"

        return result

    except Exception as e:
        print(f"  → ERROR: {e}")
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": "We were unable to process your request. A support agent will follow up shortly.",
            "justification": f"Agent error: {str(e)}",
            "request_type": request_type,
        }