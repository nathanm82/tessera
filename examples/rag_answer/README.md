# RAG answer

Shows the two halves of generation: the **prompt** a real LLM would receive (numbered,
cited context followed by the question) and the **offline answer** the default
`TemplateGenerator` produces without any model.

```bash
python examples/rag_answer/answer.py
```

To wire in a real model, implement `Generator.generate` so it sends
`build_prompt(query, context)` to your LLM of choice and returns the completion as a
`GeneratedAnswer`.
