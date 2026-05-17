import pandas as pd
from pypdf import PdfReader
from pathlib import Path


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> list[str]:
    chunks, step = [], chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if len(chunk.strip()) >= 50:
            chunks.append(chunk.strip())
    return chunks


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    records = []
    for page_num, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            print(f"  [WARN] Could not read page {page_num} of {pdf_path.name}: {e}")
            continue
        text = clean_text(raw)
        for idx, chunk in enumerate(chunk_text(text)):
            records.append({
                "source_file": pdf_path.name,
                "page":        page_num + 1,
                "chunk_id":    f"{pdf_path.stem}_p{page_num}_{idx}",
                "text":        chunk,
            })
    return records


def extract_csv_chunks(csv_path: Path) -> tuple[list[dict], pd.DataFrame]:
    df = pd.read_csv(csv_path).fillna("")

    for col in ["Quantity Ordered", "Price Each"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"Quantity Ordered", "Price Each"}.issubset(df.columns):
        df["Revenue"] = df["Quantity Ordered"] * df["Price Each"]

    records = [
        {"source_file": csv_path.name, "row": idx, "chunk_id": f"{csv_path.stem}_row_{idx}", "text": text}
        for idx, row in df.iterrows()
        if len(text := clean_text(" | ".join(f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()))) >= 20
    ]
    return records, df


def load_corpus(doc_dir: Path) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    all_chunks: list[dict] = []
    dataframes: dict[str, pd.DataFrame] = {}

    paths = sorted(doc_dir.glob("*.pdf")) + sorted(doc_dir.glob("*.csv"))
    for path in paths:
        print(f"Processing: {path.name}")
        try:
            if path.suffix.lower() == ".pdf":
                all_chunks.extend(extract_pdf_chunks(path))
            elif path.suffix.lower() == ".csv":
                chunks, df = extract_csv_chunks(path)
                all_chunks.extend(chunks)
                dataframes[path.name] = df
        except Exception as e:
            print(f"  [ERROR] Failed to process {path.name}: {e}")

    print(f"Total chunks: {len(all_chunks)}\n")
    return all_chunks, dataframes