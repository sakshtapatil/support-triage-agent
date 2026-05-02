"""
Corpus Retriever — loads all markdown files from data/ and retrieves
the most relevant documents for a given query using TF-IDF keyword search.
No external APIs needed — fully local.
"""

import os
import re
import math
from pathlib import Path
from collections import defaultdict


class CorpusRetriever:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents = []  # list of {path, domain, content, tokens}
        self._load_corpus()
        self._build_tfidf()

    def _load_corpus(self):
        """Load all .md files from the data directory."""
        domains = {
            "hackerrank": "HackerRank",
            "claude": "Claude",
            "visa": "Visa",
        }
        for domain_folder, domain_name in domains.items():
            domain_path = self.data_dir / domain_folder
            if not domain_path.exists():
                continue
            for md_file in domain_path.rglob("*.md"):
                try:
                    content = md_file.read_text(encoding="utf-8", errors="ignore")
                    if len(content.strip()) < 50:
                        continue
                    tokens = self._tokenize(content)
                    self.documents.append({
                        "path": str(md_file),
                        "domain": domain_name,
                        "content": content[:3000],  # cap at 3000 chars per doc
                        "tokens": tokens,
                        "filename": md_file.name,
                        "category": md_file.parent.name,
                    })
                except Exception:
                    continue

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [w for w in text.split() if len(w) > 2]

    def _build_tfidf(self):
        """Build IDF scores over the corpus."""
        N = len(self.documents)
        df = defaultdict(int)
        for doc in self.documents:
            for token in set(doc["tokens"]):
                df[token] += 1
        self.idf = {token: math.log(N / (1 + count)) for token, count in df.items()}

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """TF-IDF dot product score."""
        doc_tf = defaultdict(int)
        for t in doc_tokens:
            doc_tf[t] += 1
        score = 0.0
        for qt in query_tokens:
            if qt in doc_tf:
                tf = doc_tf[qt] / len(doc_tokens)
                idf = self.idf.get(qt, 0)
                score += tf * idf
        return score

    def retrieve(self, issue: str, subject: str, domain: str, top_k: int = 3) -> list[dict]:
        """Retrieve top_k most relevant documents for the given query."""
        query = f"{issue} {subject}"
        query_tokens = self._tokenize(query)

        # Filter to domain if known
        candidates = self.documents
        if domain and domain != "None":
            domain_docs = [d for d in self.documents if d["domain"] == domain]
            if domain_docs:
                candidates = domain_docs

        scored = []
        for doc in candidates:
            score = self._score(query_tokens, doc["tokens"])
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def doc_count(self) -> int:
        return len(self.documents)