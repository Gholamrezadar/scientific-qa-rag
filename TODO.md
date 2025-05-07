# Todo

- [x] Draw the diagram 
- [x] Models
  - [x] Test the keyword generation on a few models
  - [x] Test the answering on a few models
  - [x] Test a few embedding models
- [x] web.py 
  - [x] gather_and_save_document(keyword)
  - [x] fill the search_results folder with raw text from wikipedia
- [x] prompts.py
  - [x] get_search_prompt(question)
  - [x] get_retrieval_prompt(question)
  - [x] get_answer_prompt(question, context)
- [x] llm.py
  - [x] extract_keyword(question)
  - [x] embed_chunks(chunks)
  - [x] embed_question(question)
  - [x] answer_question(question, context)
- [x] rag.py
  - [x] chunk_document(document)
  - [x] store_embedded_chunks(chunks)
  - [x] retrieve_relevant_chunks(embeded_question)
- [ ] qa_rag.py
  - [ ] init model (model loading takes time. don't do it once per question)
  - [x] Read arguments (--kw-model, --embed-model, --answer-model, --kw-from-choices, --wiki-max-length, --verbose)
  - [x] gather_documents(question) // calls extract_keyword(question) and then web_gathering.py to get the text of the page for each keyword and saves them to files
  - [ ] answer_question(question, verbose=False) // does the whole pipeline
  - [ ] evaluate() // calls answer_question() on all questions and calculates the accuracy
- [ ] Demos
  - [ ] Video Demo
  - [x] Results Table (Ablations)

## Improvements

- Reranking after retrieval
- Fine-tuning the models for Multiple Choice Question Answering
- Agentic? Right now it will search the web even for a 'hi' message!

## Arguments

```text
python qa_rag.py --kw-model=gemma3:1b --embed-model=nomic-embed-text --answer-model=gemma3:1b --kw-from-choices --wiki-max-length=1000 --chunk-size=100 --chunk-overlap=50 --dataset-path=data/train_data.csv --verbose
```
