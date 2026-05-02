# 🤖 Multi-Domain Support Triage Agent

An AI-powered terminal agent that automatically triages support tickets across **HackerRank**, **Claude (Anthropic)**, and **Visa** — using a local corpus and LLM-generated responses.

Built for the HackerRank Orchestrate Hackathon (May 2026).

---

## 🏗️ Architecture

```
support_tickets.csv
        ↓
┌─────────────────────────────────────────┐
│              main.py                    │
│         (Orchestration Layer)           │
└────┬────────────┬───────────────────────┘
     ↓            ↓
classifier.py   retriever.py
(Rule-based)    (TF-IDF Search)
     ↓            ↓
     └────────────┘
           ↓
      responder.py
    (Groq LLaMA 3.3)
           ↓
      output.csv
```

### Flow per ticket:
1. **Classify** — infer domain, risk level, product area, request type using rules
2. **Escalate check** — immediately escalate dangerous/manipulative tickets
3. **Retrieve** — TF-IDF search over 773 local corpus docs to find top-3 relevant
4. **Respond** — LLM generates grounded response using only corpus context
5. **Output** — write results to CSV

---

## 📁 Project Structure

```
.
├── code/
│   ├── main.py          # Entry point
│   ├── classifier.py    # Rule-based domain/risk/type classifier
│   ├── retriever.py     # TF-IDF corpus search
│   ├── responder.py     # Groq LLM response generator
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── hackerrank/      # HackerRank support corpus
│   ├── claude/          # Claude help center corpus
│   └── visa/            # Visa support corpus
└── support_tickets/
    ├── support_tickets.csv      # Input tickets
    ├── sample_support_tickets.csv
    └── output.csv               # Agent predictions
```

---

## ⚙️ Setup

**1. Clone the repo:**
```bash
git clone https://github.com/yourusername/support-triage-agent.git
cd support-triage-agent
```

**2. Install dependencies:**
```bash
pip install -r code/requirements.txt
```

**3. Get a free Groq API key:**
- Go to [console.groq.com](https://console.groq.com)
- Create a free account → API Keys → Create Key

**4. Create `.env` in the repo root:**
```
GROQ_API_KEY=gsk_your_key_here
```

**5. Run the agent:**
```bash
cd code
python main.py
```

Output is written to `support_tickets/output.csv`.

---

## 🧠 Design Decisions

### Why TF-IDF over Vector DB?
- No external dependencies or setup required
- Fully offline — works without internet
- Fast enough for small corpus (773 docs)
- Easier to explain and debug

### Why Groq (LLaMA 3.3 70B)?
- Free tier available — no credit card needed
- Very fast inference (~1-2s per ticket)
- Strong instruction following for JSON output

### Escalation Logic
Two-stage escalation:
1. **Rule-based pre-filter** — catches prompt injection, score manipulation, identity theft, harmful requests before LLM
2. **LLM decision** — for medium-risk tickets, LLM decides based on corpus availability

### Fail-safe Design
Any error during LLM call defaults to `escalated` — never produces a hallucinated answer.

---

## 📊 Results

On 29 real support tickets:
- **23 replied** — answered from corpus
- **6 escalated** — sensitive/dangerous/out-of-scope

Correctly escalated:
- Unauthorized workspace access requests
- Score manipulation attempts
- Identity theft cases
- Harmful code requests
- Prompt injection attempts (including multilingual)

---

## 🚀 Future Improvements

- [ ] Semantic search using sentence embeddings (e.g. `sentence-transformers`)
- [ ] Confidence scoring per retrieval result
- [ ] Multi-turn conversation support
- [ ] Web UI for ticket management
- [ ] Support for more domains

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Groq API** (LLaMA 3.3 70B)
- **TF-IDF** (custom implementation, no sklearn needed)
- **python-dotenv**

---

## 📝 License

MIT