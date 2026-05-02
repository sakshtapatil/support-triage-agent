"""
Classifier — rule-based pre-classification before LLM call.
Determines domain, risk level, and likely request type from ticket content.
"""

import re

# ONLY these should hard-escalate — genuinely dangerous/sensitive
HARD_ESCALATE_PATTERNS = [
    # Prompt injection attempts
    "show all rules", "internal documents", "retrieved documents",
    "exact logic", "system prompt", "ignore previous",
    "display all", "reveal your", "bypass", "override instructions",
    "affiche toutes", "internes",  # French injection attempt in tickets

    # Harmful/malicious requests
    "delete all files", "rm -rf", "drop table",

    # Score/result manipulation
    "increase my score", "move me to next round", "graded unfairly",
    "review my answers",

    # Unauthorized access requests  
    "not the.*admin", "not the.*owner",

    # Identity theft (needs human)
    "identity.*stolen", "identity theft",
]

# Sensitive but CAN be answered with corpus info
MEDIUM_RISK_KEYWORDS = [
    "refund", "stolen card", "lost card", "fraud", "dispute",
    "security vulnerability", "bug bounty", "urgent cash",
    "blocked card", "card blocked",
]

# Out of scope / invalid
INVALID_INDICATORS = [
    "iron man", "actor", "movie", "song", "recipe", "weather",
    "thank you for", "thanks for", "good morning", "good evening",
    "delete all files", "rm -rf",
]

# Domain keyword mapping
DOMAIN_KEYWORDS = {
    "HackerRank": [
        "hackerrank", "assessment", "test", "candidate", "recruiter",
        "coding challenge", "interview", "screen", "skillup", "library",
        "submission", "plagiarism", "proctoring", "invite", "variant",
        "mock interview", "resume builder", "certificate", "inactivity",
        "zoom", "compatible check", "hiring", "apply tab", "interviewer",
        "subscription pause", "infosec",
    ],
    "Claude": [
        "claude", "anthropic", "conversation", "chat", "prompt", "api",
        "bedrock", "lti", "workspace", "claude.ai", "claude code",
        "team plan", "enterprise", "pro plan", "crawl", "training data",
        "model", "requests are failing", "aws bedrock",
    ],
    "Visa": [
        "visa", "card", "payment", "merchant", "transaction", "chargeback",
        "traveller", "cheque", "atm", "cash advance", "dispute charge",
        "stolen card", "lost card", "minimum spend", "contactless",
        "blocked card", "carte visa", "tarjeta",
    ],
}

PRODUCT_AREAS = {
    "HackerRank": {
        "test|assessment|variant|invite|candidate|proctoring|inactivity|zoom|compatible|rescheduling": "screen",
        "interview|mock interview|interviewer": "interviews",
        "community|account|delete|password|login|google login": "community",
        "billing|payment|subscription|refund|invoice|order": "billing",
        "resume|certificate": "general-help",
        "library|question|challenge|submission|apply tab|practice": "library",
        "integration|infosec|security|sso": "integrations",
        "skillup|practice|learn": "skillup",
        "hiring|recruiter|employee|user|remove|seat": "settings",
    },
    "Claude": {
        "privacy|delete|conversation|data|crawl|training": "privacy-and-legal",
        "api|bedrock|aws|integration|requests.*failing": "claude-api-and-console",
        "team|enterprise|workspace|admin|seat": "team-and-enterprise-plans",
        "pro|max|subscription|billing|plan": "pro-and-max-plans",
        "lti|education|professor|student": "claude-for-education",
        "bug|vulnerability|security|bounty": "safeguards",
        "mobile|app|ios|android": "claude-mobile-apps",
        "desktop": "claude-desktop",
        "code|coding": "claude-code",
    },
    "Visa": {
        "lost|stolen|block|card": "general_support",
        "dispute|chargeback|refund|merchant|wrong product": "dispute_resolution",
        "traveller|cheque": "travel_support",
        "identity|theft|fraud": "fraud_support",
        "cash|atm|withdraw|urgent cash": "general_support",
        "minimum|spend|surcharge": "general_support",
    },
}


def infer_domain(issue: str, subject: str, company: str) -> str:
    if company and company.strip() not in ("", "None"):
        return company.strip()
    text = f"{issue} {subject}".lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "None"


def infer_product_area(issue: str, subject: str, domain: str) -> str:
    text = f"{issue} {subject}".lower()
    if domain not in PRODUCT_AREAS:
        return "general"
    for pattern, area in PRODUCT_AREAS[domain].items():
        if re.search(pattern, text):
            return area
    return "general"


def assess_risk(issue: str, subject: str) -> str:
    """
    high   = must escalate (manipulation, injection, identity theft)
    medium = sensitive but corpus may answer (fraud info, card issues)
    low    = routine question
    """
    text = f"{issue} {subject}".lower()
    for pattern in HARD_ESCALATE_PATTERNS:
        if re.search(pattern, text):
            return "high"
    for kw in MEDIUM_RISK_KEYWORDS:
        if kw in text:
            return "medium"
    return "low"


def infer_request_type(issue: str, subject: str) -> str:
    text = f"{issue} {subject}".lower()
    for kw in INVALID_INDICATORS:
        if kw in text:
            return "invalid"
    if any(w in text for w in [
        "not working", "broken", "down", "error", "bug", "failing",
        "issue", "crash", "blocker", "stopped", "can not", "cannot",
        "unable", "doesn't work", "not responding"
    ]):
        return "bug"
    if any(w in text for w in [
        "feature", "suggestion", "would be nice", "request", "add support",
        "improve", "extend", "can you add", "please add"
    ]):
        return "feature_request"
    return "product_issue"


def classify_ticket(issue: str, subject: str, company: str) -> dict:
    domain = infer_domain(issue, subject, company)
    risk = assess_risk(issue, subject)
    request_type = infer_request_type(issue, subject)
    product_area = infer_product_area(issue, subject, domain)

    return {
        "domain": domain,
        "risk": risk,
        "request_type": request_type,
        "product_area": product_area,
    }