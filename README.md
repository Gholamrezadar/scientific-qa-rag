# Scientific Question Answering using RAG

Answering Multiple Choice Scientific Questions using RAG + Internet Search.

## Diagram

![diagram](demos/qa_rag_diagram.png)

## Usage

```bash
python qa_rag.py
```

## Dataset

50 multiple choice questions about various scientific topics. [train_data.csv](data/train_data.csv) and [test_data.csv](data/test_data.csv)

## Results

| Name                   | avg_accuracy | acc1 | acc2 | params | speed  | std |
| ---------------------- | ------------ | ---- | ---- | ------ | ------ | --- |
| Random                 | 20%          | 20%  | 20%    | 0      | -      | 0%  |
| Gemma3:1b (no-context) | 39%          | 34%  | 44%  | 1B     | 4s/it  | 5%  |
| phi4:14b (no-context)  | 78%          | 78%  | -    | 14B    | 40s/it | 0%  |

## Credits

By Gholamreza Dar 2025
