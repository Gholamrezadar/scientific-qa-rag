# Scientific Question Answering using RAG

Answering Multiple Choice Scientific Questions using RAG + Internet Search.

## Diagram

![diagram](demos/qa_rag_diagram.png)

## Usage

```bash
python qa_rag.py
```

## Results

| Name                           | avg_accuracy | acc_min | acc_max | params | speed   |
| ------------------------------ | ------------ | ------- | ------- | ------ | ------- |
| Random                         | 20%          | 20%     | 20%     | 0      | -       |
| Gemma3:1b (no-context)         | 39%          | 34%     | 44%     | 1B     | 4s/it   |
| gemma3:12b-it-qat (no-context) | 78%          | -       | -       | 12B    | 26s/it  |
| phi4:14b (no-context)          | 78%          | -       | -       | 14B    | 40s/it  |
| GPT-4o (no-context)            | 88%          | -       | -       | 600B+  | 16s/it* |

`* GPT-4o was run using OpenAI's Infrastructure. Other models were run on a Colab T4 GPU.`

## Dataset

50 multiple choice questions about various scientific topics. [train_data.csv](data/train_data.csv) and [test_data.csv](data/test_data.csv)

Example Prompt:

```text
Answer this multiple choice question using the context below.
You should be very careful to answer the question correctly. So think about the problem step by step.

Context: [INSERT CONTEXT HERE]

Question: What is the proposed name for the field that is responsible for cosmic inflation and the metric expansion of space?

Choices:
Choice A) Inflaton
Choice B) Quanta
Choice C) Scalar
Choice D) Metric
Choice E) Conformal cyclic cosmology

It's ok to talk and think about the problem but the last line of your answer should be the string 'Answer:' followed by either A, B, C, D, or E.
Example answer:
'Based on the context, the best answer would be Oxygen and because Choice A) Oxygen,
Answer: A'
```

## Credits

- [Langchain Ollama docs](https://python.langchain.com/docs/integrations/providers/ollama/)
- [Langchain Ollama Embedding docs](https://python.langchain.com/docs/integrations/text_embedding/ollama/)
- [Langchain Ollama Embedding API reference](https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html#langchain_ollama.embeddings.OllamaEmbeddings)
- [Langchain OllamaLLM API reference](https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html#langchain_ollama.llms.OllamaLLM)
- [Langchain Chroma docs](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [Langchain Chroma API reference](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html)

By Gholamreza Dar 2025
