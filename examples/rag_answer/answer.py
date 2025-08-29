"""Inspect the grounded prompt and the templated answer the pipeline builds.

This is the seam where you would call a real LLM: ``TemplateGenerator.prompt`` returns
exactly the text you would send to a model, while ``answer`` shows the offline,
deterministic fallback.

Run with::

    python examples/rag_answer/answer.py
"""

from tessera import Document, PipelineConfig, RagPipeline
from tessera.generation.template import TemplateGenerator

DOCUMENTS = [
    Document(
        id="d1",
        text="Retrieval-augmented generation grounds a model's answer in retrieved context.",
    ),
    Document(id="d2", text="A dual encoder maps text and images into one shared vector space."),
    Document(id="d3", text="Fusion blends ranked results from several retrievers into one list."),
]


def main() -> None:
    pipeline = RagPipeline(PipelineConfig(top_k=3))
    pipeline.add_documents(DOCUMENTS)

    query = "what grounds a model's answer in retrieved context?"
    results = pipeline.retrieve(query)

    generator = TemplateGenerator()
    print("=== Prompt an LLM would receive ===")
    print(generator.prompt(query, results))
    print("\n=== Offline templated answer ===")
    print(pipeline.answer(query).text)


if __name__ == "__main__":
    main()
