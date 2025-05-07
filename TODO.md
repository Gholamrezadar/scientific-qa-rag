# Todo

- [x] Draw the diagram 
- [ ] Models
  - [ ] Test the keyword generation on a few models
  - [ ] Test the answering on a few models
  - [ ] Test a few embedding models
- [ ] web.py 
  - [ ] gather_and_save_document(keyword)
  - [ ] fill the search_results folder with raw text from wikipedia
- [ ] prompts.py
  - [ ] get_search_prompt(question)
  - [ ] get_retrieval_prompt(question)
  - [ ] get_answer_prompt(question, context)
- [ ] llm.py
  - [ ] extract_keyword(question)
  - [ ] embed_chunks(chunks)
  - [ ] embed_question(question)
  - [ ] answer_question(question, context)
- [ ] rag.py
  - [ ] chunk_document(document)
  - [ ] store_embedded_chunks(chunks)
  - [ ] retrieve_relevant_chunks(embeded_question)
- [ ] qa_rag.py
    - [ ] init model (model loading takes time. don't do it once per question)
    - [ ] Read arguments (--kw-model, --embed-model, --answer-model, --kw-from-choices, --wiki-max-length, --verbose)
    - [ ] gather_documents(question) // calls extract_keyword(question) and then web_gathering.py to get the text of the page for each keyword and saves them to files 
    - [ ] answer_question(question, verbose=False) // does the whole pipeline
    - [ ] evaluate() // calls answer_question() on all questions and calculates the accuracy
- [ ] Demos
  - [ ] Video Demo
  - [ ] Results Table (Ablations)


## Improvements

- Reranking after retrieval
- Fine-tuning the models for Multiple Choice Question Answering
- Agentic? Right now it will search the web even for a 'hi' message!

## Arguments

python qa_rag.py --kw-model=gemma3:1b --embed-model=nomic-embed-text --answer-model=gemma3:1b --kw-from-choices --wiki-max-length=1000 --chunk-size=100 --chunk-overlap=50 --dataset-path=data/train_data.csv --verbose
-  