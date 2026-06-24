"""Index a tiny text corpus, then retrieve and answer over it.

Run with::

    python examples/quickstart/quickstart.py
"""

from tessera import Document, RagPipeline

DOCUMENTS = [
    Document(id="reef", text="The Great Barrier Reef is the world's largest coral reef system."),
    Document(id="python", text="Python is a high-level, general-purpose programming language."),
    Document(id="coral", text="Coral reefs support about a quarter of all marine species."),
]


def main() -> None:
    pipeline = RagPipeline()
    pipeline.add_documents(DOCUMENTS)

    query = "tell me about coral reefs"
    print("Top matches:")
    for result in pipeline.retrieve(query, top_k=2):
        print(f"  {result.score:.3f}  {result.chunk.doc_id}: {result.chunk.content}")

    print("\nAnswer:")
    print(pipeline.answer(query, top_k=2).text)


if __name__ == "__main__":
    main()
