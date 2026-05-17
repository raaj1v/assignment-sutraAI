from pathlib import Path
import re
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# =========================================================
# CONFIG
# =========================================================

DOC_DIR = Path("./dataDocs")

LLM_CLIENT = OpenAI(
    base_url="http://localhost:4141/v1",
    api_key="dummy"
)
MODEL_NAME = "gpt-5-mini"

EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

CONFIDENCE_THRESHOLD = 0.35
TOP_K = 3

ANALYTICAL_KEYWORDS = {
    "top", "highest", "lowest", "average", "avg",
    "total", "sum", "count", "rank", "minimum",
    "maximum", "max", "min", "trend", "compare",
    "best", "worst", "most", "least", "revenue",
    "sales", "quantity", "orders", "product"
}

INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"you are now",
    r"forget (everything|your instructions)",
    r"system\s*:",
    r"<\s*/?system\s*>",
    r"disregard",
]

AMBIGUOUS_PATTERNS = [
    (r"\bthe report\b",              "Which report are you referring to?"),
    (r"\bthe process\b",             "Which process do you mean?"),
    (r"\bbad\b|\bissue\b|\bproblem\b","Can you be more specific about what aspect concerns you?"),
    (r"\brecently\b|\blast\b",       "What time period are you asking about?"),
    (r"\bthey\b|\bthem\b|\bit\b",    "Could you clarify who or what you are referring to?"),
]

# =========================================================
# CONVERSATION MEMORY (in-session)
# =========================================================

conversation_history = []
MAX_HISTORY_TURNS = 6

# =========================================================
# HELPERS
# =========================================================

def clean_text(text):
    return " ".join(str(text).split())


def chunk_text(text, chunk_size=1500, overlap=300):
    chunks, step = [], chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if len(chunk.strip()) >= 50:
            chunks.append(chunk.strip())
    return chunks


def _update_history(question, answer):
    conversation_history.append({"role": "user",      "content": question})
    conversation_history.append({"role": "assistant", "content": answer})
    if len(conversation_history) > MAX_HISTORY_TURNS * 2:
        del conversation_history[:2]

# =========================================================
# GUARDRAIL 1 — prompt injection detection
# Why: user input flows directly into LLM prompts; crafted
#      queries can override system instructions.
# Mitigates: instruction override, data exfiltration.
# Limitation: regex-based; novel phrasing may bypass it.
# =========================================================

def detect_injection(text):
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)

# =========================================================
# PARSERS
# =========================================================

def extract_pdf_chunks(pdf_path):
    reader = PdfReader(str(pdf_path))
    records = []
    for page_num, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        for idx, chunk in enumerate(chunk_text(text)):
            records.append({
                "source_file": pdf_path.name,
                "page":        page_num + 1,
                "chunk_id":    f"{pdf_path.stem}_p{page_num}_{idx}",
                "text":        chunk,
            })
    return records


def extract_csv_chunks(csv_path):
    df = pd.read_csv(csv_path).fillna("")

    # Coerce numeric columns
    for col in ["Quantity Ordered", "Price Each"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived column for analytics
    if "Quantity Ordered" in df.columns and "Price Each" in df.columns:
        df["Revenue"] = df["Quantity Ordered"] * df["Price Each"]

    records = []
    for idx, row in df.iterrows():
        text = " | ".join(
            f"{col}: {row[col]}"
            for col in df.columns
            if str(row[col]).strip()
        )
        text = clean_text(text)
        if len(text) >= 20:
            records.append({
                "source_file": csv_path.name,
                "row":         idx,
                "chunk_id":    f"{csv_path.stem}_row_{idx}",
                "text":        text,
            })
    return records, df

# =========================================================
# BUILD CORPUS
# =========================================================

all_chunks = []
dataframes = {}

for path in sorted(DOC_DIR.glob("*.pdf")) + sorted(DOC_DIR.glob("*.csv")):
    print(f"Processing: {path.name}")
    if path.suffix.lower() == ".pdf":
        all_chunks.extend(extract_pdf_chunks(path))
    elif path.suffix.lower() == ".csv":
        chunks, df = extract_csv_chunks(path)
        all_chunks.extend(chunks)
        dataframes[path.name] = df
    else:
        print(f"  Skipping: {path.name}")

print(f"\nTotal chunks: {len(all_chunks)}")

# =========================================================
# VECTOR STORE  (embed once, reuse across runs)
# =========================================================

chroma_client = chromadb.PersistentClient("./vectordb")
collection    = chroma_client.get_or_create_collection("enterprise_rag")

existing_ids = set(collection.get()["ids"])
new_chunks   = [c for c in all_chunks if c["chunk_id"] not in existing_ids]

if new_chunks:
    print(f"Embedding {len(new_chunks)} new chunks …")
    for chunk in new_chunks:
        embedding = EMBED_MODEL.encode(
            chunk["text"], normalize_embeddings=True
        ).tolist()

        metadata = {
            "source_file": str(chunk["source_file"]),
            "chunk_id":    str(chunk["chunk_id"]),
        }
        if chunk.get("page") is not None:
            metadata["page"] = int(chunk["page"])
        if chunk.get("row") is not None:
            metadata["row"] = int(chunk["row"])

        collection.add(
            ids=[chunk["chunk_id"]],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[metadata],
        )
    print("Embedding complete.")
else:
    print("All chunks already embedded — skipping.")

print("Vector store ready.\n")

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(query, top_k=TOP_K):
    q_emb = EMBED_MODEL.encode(
        query, normalize_embeddings=True
    ).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

# =========================================================
# GUARDRAIL 2 — retrieval confidence threshold
# Why: low similarity means no real evidence; LLM hallucinates.
# Mitigates: fabricated answers on out-of-scope questions.
# Limitation: cosine distance is a proxy — semantically close
#             but factually different chunks can still pass.
# =========================================================

def confidence_from_distance(distance):
    # ChromaDB cosine distance is in [0, 2]; map to [0, 1]
    return 1.0 - (distance / 2.0)

# =========================================================
# PART 3 — structured / analytical query handler
# Columns: Order ID, Product, Quantity Ordered,
#          Price Each, Order Date, Purchase Address, Revenue. (this has been added during CSV parsing specially calculated for analytical queries pertaining to sales data)
# =========================================================

def is_analytical(query):
    tokens = set(query.lower().split())
    return bool(tokens & ANALYTICAL_KEYWORDS)


def run_analytical_query(query, dfs):
    q = query.lower()
    results = []

    for fname, df in dfs.items():
        if df.empty:
            continue

        # ── top N products by revenue / quantity ──────────────
        top_match = re.search(r"top\s+(\d+)", q)
        if top_match:
            n = int(top_match.group(1))
            if "revenue" in q and "Revenue" in df.columns:
                tbl = (
                    df.groupby("Product")["Revenue"]
                    .sum().nlargest(n)
                    .reset_index()
                    .rename(columns={"Revenue": "Total Revenue ($)"})
                )
                tbl["Total Revenue ($)"] = tbl["Total Revenue ($)"].round(2)
            elif "quantity" in q and "Quantity Ordered" in df.columns:
                tbl = (
                    df.groupby("Product")["Quantity Ordered"]
                    .sum().nlargest(n)
                    .reset_index()
                    .rename(columns={"Quantity Ordered": "Total Qty"})
                )
            else:
                # default: revenue
                tbl = (
                    df.groupby("Product")["Revenue"]
                    .sum().nlargest(n)
                    .reset_index()
                    .rename(columns={"Revenue": "Total Revenue ($)"})
                )
                tbl["Total Revenue ($)"] = tbl["Total Revenue ($)"].round(2)
            results.append(f"[{fname}]\n{tbl.to_string(index=False)}")
            continue

        # ── total revenue ──────────────────────────────────────
        if any(kw in q for kw in ("total revenue", "total sales")):
            if "Revenue" in df.columns:
                total = df["Revenue"].sum()
                results.append(f"[{fname}] Total revenue: ${total:,.2f}")
            continue

        # ── average order value ────────────────────────────────
        if any(kw in q for kw in ("average", "avg", "mean")):
            if "Revenue" in df.columns:
                avg = df["Revenue"].mean()
                results.append(f"[{fname}] Average order revenue: ${avg:,.2f}")
            if "Quantity Ordered" in df.columns:
                avg_q = df["Quantity Ordered"].mean()
                results.append(f"[{fname}] Average quantity ordered: {avg_q:.1f}")
            continue

        # ── best / worst selling product ───────────────────────
        if any(kw in q for kw in ("best", "highest", "most")):
            if "Revenue" in df.columns:
                top = df.groupby("Product")["Revenue"].sum().idxmax()
                val = df.groupby("Product")["Revenue"].sum().max()
                results.append(
                    f"[{fname}] Best selling product: {top} (${val:,.2f})"
                )
            continue

        if any(kw in q for kw in ("worst", "lowest", "least")):
            if "Revenue" in df.columns:
                bot = df.groupby("Product")["Revenue"].sum().idxmin()
                val = df.groupby("Product")["Revenue"].sum().min()
                results.append(
                    f"[{fname}] Lowest revenue product: {bot} (${val:,.2f})"
                )
            continue

        # ── orders count ───────────────────────────────────────
        if "count" in q or "how many orders" in q:
            results.append(f"[{fname}] Total orders: {len(df):,}")
            continue

        # ── product lookup ─────────────────────────────────────
        if "product" in q:
            products = df["Product"].unique().tolist()
            results.append(
                f"[{fname}] Products in dataset ({len(products)}):\n"
                + ", ".join(products[:20])
                + ("…" if len(products) > 20 else "")
            )
            continue

    return "\n\n".join(results) if results else None

# =========================================================
# PART 2 — ambiguity detection
# =========================================================

def detect_ambiguity(query):
    q = query.lower()
    for pattern, question in AMBIGUOUS_PATTERNS:
        if re.search(pattern, q):
            return question
    return None

# =========================================================
# CONVERSATION — query rewriter
# =========================================================

def rewrite_query_with_context(question):
    """
    Rewrites a follow-up question into a self-contained search
    query using recent conversation history.
    E.g. "What about step 2?" → "What is step 2 of the HHS
    strategic goal review process described earlier?"
    """
    if not conversation_history:
        return question

    recent = conversation_history[-4:]
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in recent
    )

    rewrite_prompt = f"""Given this conversation history:
{history_text}

Rewrite the following question as a fully self-contained search query.
Output ONLY the rewritten query, nothing else.

Question: {question}
Rewritten query:"""

    response = LLM_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(question, retrieved_docs):
    context = ""
    for r in retrieved_docs:
        meta = r["metadata"]
        context += f"\nSOURCE: {meta['source_file']}\n{r['text']}\n"

    history_text = ""
    if conversation_history:
        history_text = "\nCONVERSATION SO FAR:\n"
        for m in conversation_history[-MAX_HISTORY_TURNS:]:
            history_text += f"{m['role'].capitalize()}: {m['content']}\n"

    return f"""You are a grounded enterprise assistant.

Rules:
- Answer ONLY from the provided context below.
- If the context does not contain enough information, say so explicitly.
- Always cite the source file(s) you used.
- Never fabricate data, names, or figures.
- Use the conversation history to understand follow-up questions.
{history_text}
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def generate_answer(prompt):
    response = LLM_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# =========================================================
# MAIN RAG FUNCTION
# =========================================================

def ask_rag(question):
    # Guardrail 1 — injection check
    if detect_injection(question):
        return {
            "answer":     "Your query contains patterns that look like prompt injection. Please rephrase.",
            "confidence": 0.0,
            "sources":    [],
            "flag":       "injection_detected",
        }

    # Part 2A — ambiguity check
    clarification = detect_ambiguity(question)
    if clarification:
        return {
            "answer":     f"Your question seems ambiguous. {clarification}",
            "confidence": None,
            "sources":    [],
            "flag":       "clarification_needed",
        }

    # Part 3 — analytical routing (CSV only)
    if is_analytical(question) and dataframes:
        analytical_result = run_analytical_query(question, dataframes)
        if analytical_result:
            _update_history(question, analytical_result)
            return {
                "answer":     analytical_result,
                "confidence": 1.0,
                "sources":    [{"source_file": f} for f in dataframes],
                "flag":       "analytical",
            }

    # Rewrite query using conversation context before retrieval
    search_query = rewrite_query_with_context(question)

    # Standard RAG path
    retrieved_docs = retrieve(search_query)
    top_confidence = confidence_from_distance(retrieved_docs[0]["distance"])

    # Guardrail 2 — confidence threshold
    if top_confidence < CONFIDENCE_THRESHOLD:
        return {
            "answer": (
                "I could not find sufficient evidence in the documents to answer this. "
                f"Best match confidence: {top_confidence:.2f}. "
                "Try rephrasing or check that the relevant document is loaded."
            ),
            "confidence": top_confidence,
            "sources":    [],
            "flag":       "low_confidence",
        }

    prompt = build_prompt(question, retrieved_docs)
    answer = generate_answer(prompt)

    _update_history(question, answer)

    sources = [
        {
            "source_file": r["metadata"]["source_file"],
            "page":        r["metadata"].get("page"),
            "row":         r["metadata"].get("row"),
            "snippet":     r["text"][:200],
            "confidence":  round(confidence_from_distance(r["distance"]), 2),
        }
        for r in retrieved_docs
    ]

    return {
        "answer":     answer,
        "confidence": round(top_confidence, 2),
        "sources":    sources,
        "flag":       "ok",
    }

# =========================================================
# PART 5 — lightweight evaluation
# =========================================================

def evaluate(test_cases):
    print(f"\n{'='*60}")
    print(f"{'EVALUATION REPORT':^60}")
    print(f"{'='*60}")

    scores = []
    for tc in test_cases:
        result = ask_rag(tc["question"])
        answer = result["answer"].lower()
        keywords = tc.get("expected_keywords", [])

        kw_hits  = sum(1 for kw in keywords if kw.lower() in answer)
        kw_score = kw_hits / len(keywords) if keywords else 1.0

        source_match = None
        if tc.get("expected_source"):
            cited = any(
                tc["expected_source"] in str(s.get("source_file", ""))
                for s in result.get("sources", [])
            )
            source_match = int(cited)

        scores.append({
            "question":      tc["question"][:60],
            "keyword_score": round(kw_score, 2),
            "source_match":  source_match,
            "confidence":    result.get("confidence"),
            "flag":          result.get("flag"),
        })

        print(f"\nQ : {tc['question'][:60]}")
        print(f"    keyword_score : {kw_score:.2f}  ({kw_hits}/{len(keywords)} keywords found)")
        if source_match is not None:
            print(f"    source_match  : {'yes' if source_match else 'NO'}")
        print(f"    confidence    : {result.get('confidence')}")
        print(f"    flag          : {result.get('flag')}")

    avg_kw = np.mean([s["keyword_score"] for s in scores])
    print(f"\n{'─'*60}")
    print(f"Average keyword score : {avg_kw:.2f}")
    print(f"{'='*60}\n")
    return scores

# =========================================================
# INTERACTIVE Q&A LOOP
# =========================================================

if __name__ == "__main__":
    print("Enterprise RAG assistant")
    print("Commands: 'quit' to exit | 'eval' to run evaluation | 'history' to show memory\n")

    EVAL_CASES = [
        {
            "question":          "What were HHS's strategic goals for 2022?",
            "expected_keywords": ["strategic", "goal", "health"],
            "expected_source":   "HHS2022.pdf",
        },
        {
            "question":          "What is the top 5 products by revenue?",
            "expected_keywords": ["product", "revenue"],
            "expected_source":   "Sales_April_2019.csv",
        },
        {
            "question":          "What is NARA's performance target?",
            "expected_keywords": ["target", "performance", "nara"],
            "expected_source":   "NARA2022.pdf",
        },
        {
            "question":          "What is the total revenue?",
            "expected_keywords": ["revenue", "$"],
            "expected_source":   "Sales_April_2019.csv",
        },
    ]

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() == "quit":
            break

        if question.lower() == "eval":
            evaluate(EVAL_CASES)
            continue

        if question.lower() == "history":
            if not conversation_history:
                print("No history yet.\n")
            else:
                for m in conversation_history:
                    print(f"  {m['role'].capitalize()}: {m['content'][:120]}")
                print()
            continue

        result = ask_rag(question)

        print(f"\nAnswer     : {result['answer']}")
        print(f"Confidence : {result.get('confidence')}")
        print(f"Flag       : {result.get('flag')}")

        if result.get("sources"):
            print("Sources    :")
            for s in result["sources"]:
                loc = f"page {s['page']}" if s.get("page") else f"row {s.get('row', '?')}"
                print(f"  - {s['source_file']} ({loc}) | conf: {s.get('confidence')}")
                print(f"    \"{s['snippet']}...\"")
        print()