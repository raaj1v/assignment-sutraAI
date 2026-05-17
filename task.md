## Overview

This assignment evaluates your ability to design and implement a lightweight, reliable AI assistant under realistic constraints. It is intentionally scoped to **3–4 hours** and does not require production polish.

## What we’re evaluating

- Problem understanding and assumptions
- Practical AI/LLM solution design (tradeoffs, failure modes)
- Retrieval quality and grounding (citations/snippets)
- Hallucination and ambiguity handling
- Guardrails and reliability thinking
- Evaluation approach and quality measurement
- Overall engineering judgment

---

## Problem context

Organizations often have institutional knowledge scattered across:

- Documents (SOPs, process docs, policies)
- PDFs and reports
- Emails and communication samples
- Spreadsheets (operational and KPI data)

As a result, employees spend time:

- Searching for information
- Interpreting operational data
- Understanding processes and reports
- Finding the correct guidance quickly

Your goal: build a small assistant that answers questions **grounded in provided documents** and handles uncertainty safely.

---

## Dataset / input documents

Use **3–5** sample business documents of your choice. Examples:

- SOP / policy documents
- Process documents
- PDF reports
- Emails / communication samples
- Excel/CSV operational reports

You may create dummy/sample documents if needed.

---

## Requirements

### Part 1 — Build a lightweight RAG assistant

Build a simple assistant that can:

- Accept user questions
- Retrieve relevant information from the uploaded documents
- Generate **grounded** answers
- Provide **source references/snippets** used in the answer
- Handle unknown answers gracefully (don’t fabricate)

**Example questions**

- “What is the escalation process for delayed shipments?”
- “Explain the inventory aging KPI.”
- “What is the approval workflow for procurement requests?”

---

### Part 2 — Ambiguity & hallucination handling (choose at least one)

Implement **at least one** of the following:

**Option A — Clarification flow**  

When a question is ambiguous, ask clarifying questions *before* answering.

Example prompt:

> “Why is the report bad?”
> 

Expected behavior:

- Ask what report, what “bad” means (accuracy? completeness? performance?), timeframe, audience, etc.

**Option B — Confidence / uncertainty handling**  

When the evidence is weak or retrieval confidence is low:

- Respond with uncertainty
- Avoid unsupported claims
- Explicitly state what is missing
- Offer what you *can* answer from available sources

---

### Part 3 — Structured data understanding

Include at least one structured file (Excel/CSV). Demonstrate:

- Reading structured data
- Answering simple analytical questions

**Example questions**

- “Which branch has the highest sales?”
- “What is the average inventory aging?”
- “Show top 5 SKUs by aging days.”

---

### Part 4 — Guardrails & reliability (implement at least two)

Implement **at least 2 guardrails**. Examples:

- Refuse/abstain when sources don’t support an answer
- Minimum retrieval threshold before answering
- Enforce “citation-only” answers for factual claims
- Prompt injection detection/mitigation
- Sensitive information masking
- Basic access restriction simulation

In your README, briefly explain for each guardrail:

- Why it’s needed
- What risk it mitigates
- Known limitations

---

### Part 5 — Evaluation & quality assessment

Design a lightweight evaluation approach to assess answer quality. Examples:

- Retrieval relevance checks
- Groundedness / citation coverage checks
- Hallucination detection heuristics
- Human evaluation rubric
- Confidence scoring

A simple methodology or small evaluation script is sufficient.

---

## Deliverables

### 1) Source code

- GitHub repository **or** ZIP file

### 2) README (required)

Include:

- Setup instructions
- Architecture overview (diagram optional)
- Key design choices and tradeoffs
- Assumptions
- Guardrails implemented
- Evaluation approach
- Limitations
- Future improvements

### 3) Demo

- Demo video link
- Any sample documents used

---

## Constraints

- Target effort: **3–4 hours**
- Keep the implementation lightweight
- Prioritize correctness, grounding, and safe failure modes over polish

---

## What we are NOT looking for

You do **not** need:

- A fancy UI
- Complex infrastructure
- Large-scale deployment
- Fine-tuning

We prefer:

- Good judgment
- Practical implementation
- Clear reasoning
- Reliable AI behavior