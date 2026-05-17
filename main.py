from pipeline import RAGPipeline

EVAL_CASES = [
    {
        "question": "What were HHS's strategic goals for 2022?",
        "expected_keywords": ["strategic", "goal", "health"],
        "expected_source": "HHS2022.pdf",
    },
    {
        "question": "What is the top 5 products by revenue?",
        "expected_keywords": ["product", "revenue"],
        "expected_source": "Sales_April_2019.csv",
    },
    {
        "question": "What is NARA's performance target?",
        "expected_keywords": ["target", "performance", "nara"],
        "expected_source": "NARA2022.pdf",
    },
    {
        "question": "What is the total revenue?",
        "expected_keywords": ["revenue", "$"],
        "expected_source": "Sales_April_2019.csv",
    },
]


def main():
    rag = RAGPipeline()

    print("Enterprise RAG assistant")
    print("Commands: 'quit' | 'eval' | 'history'\n")

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
            rag.evaluate(EVAL_CASES)
            continue
        if question.lower() == "history":
            if not rag.conversation_history:
                print("No history yet.\n")
            else:
                for m in rag.conversation_history:
                    print(f"  {m['role'].capitalize()}: {m['content'][:120]}")
                print()
            continue

        result = rag.ask(question)
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


if __name__ == "__main__":
    main()