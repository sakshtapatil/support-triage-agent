#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Handles HackerRank, Claude, and Visa support tickets
"""

import os
import sys
import csv
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from retriever import CorpusRetriever
from classifier import classify_ticket
from responder import generate_response

# Seed for determinism
random.seed(42)

# Paths (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
TICKETS_PATH = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH = REPO_ROOT / "support_tickets" / "output.csv"

# Load .env from repo root
load_dotenv(REPO_ROOT / ".env")

OUTPUT_COLUMNS = ["Issue", "Subject", "Company", "status", "product_area", "response", "justification", "request_type"]


def load_tickets(path: Path) -> list:
    tickets = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(row)
    return tickets


def write_output(results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Output written to {path}")


def process_ticket(ticket: dict, retriever: CorpusRetriever, idx: int, total: int) -> dict:
    issue = ticket.get("Issue", "").strip()
    subject = ticket.get("Subject", "").strip()
    company = ticket.get("Company", "").strip()

    print(f"\n[{idx+1}/{total}] Processing: {issue[:80]}...")

    # Step 1: Classify
    classification = classify_ticket(issue, subject, company)
    print(f"  → Domain: {classification['domain']} | Risk: {classification['risk']} | Type: {classification['request_type']}")

    # Step 2: Retrieve relevant corpus docs
    docs = retriever.retrieve(issue, subject, classification['domain'], top_k=3)
    print(f"  → Retrieved {len(docs)} corpus docs")

    # Step 3: Generate response
    result = generate_response(issue, subject, company, classification, docs)
    print(f"  → Status: {result['status']} | Product Area: {result['product_area']}")

    return {
        "Issue": issue,
        "Subject": subject,
        "Company": company,
        "status": result["status"],
        "product_area": result["product_area"],
        "response": result["response"],
        "justification": result["justification"],
        "request_type": result["request_type"],
    }


def main():
    print("=" * 60)
    print("  Multi-Domain Support Triage Agent")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not set")
        print("   Add it to your .env file in the repo root:")
        print("   GROQ_API_KEY=gsk_your_key_here")
        sys.exit(1)

    print(f"\n📂 Loading corpus from {DATA_DIR}...")
    retriever = CorpusRetriever(DATA_DIR)
    print(f"   Loaded {retriever.doc_count()} documents")

    print(f"\n📋 Loading tickets from {TICKETS_PATH}...")
    tickets = load_tickets(TICKETS_PATH)
    print(f"   Found {len(tickets)} tickets")

    results = []
    for idx, ticket in enumerate(tickets):
        result = process_ticket(ticket, retriever, idx, len(tickets))
        results.append(result)

    write_output(results, OUTPUT_PATH)

    # Summary
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"\n📊 Summary: {replied} replied, {escalated} escalated out of {len(results)} tickets")


if __name__ == "__main__":
    main()