import argparse
import logging
import sys
from os import makedirs
import time

from typing import List

import pandas as pd
from tqdm.auto import tqdm

from src.prompts import get_keyword_generation_prompt
from src.llm import generate_keywords 


def setup_logging(verbose: bool):

    # Mute LangChain and other libraries
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langchain.callbacks.manager").setLevel(logging.ERROR)
    logging.getLogger("langchain_core").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # if httpx is noisy too

    handler = logging.StreamHandler(sys.stdout)
    handler.flush = sys.stdout.flush  # force flushing

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        handlers=[handler],
        level=level
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Answer Multiple Choice Scientific Questions using RAG and Internet Search", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--num-samples", default=-1, type=int, help="Limit the number of questions for faster debugging. Set to -1 to use all questions.")
    parser.add_argument("--kw-model", default='gemma3:1b', help="Model used for keyword extraction")
    parser.add_argument("--answer-model", default='gemma3:1b', help="Model used to generate answers")
    parser.add_argument("--embed-model", default='nomic-embed-text', help="Model used for embedding")
    parser.add_argument("--kw-from-choices", action="store_true", help="Extract keywords from the choices too (if false, only extract from the question)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of chunks for retrieval")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--load-kw", action="store_true", help="If set, load keywords from file.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    # 1. Load dataset
    # 2. Create kw prompt for each question
    # 3. Genrate kw response from model
    # 4. Extract kws
    # 5. Download docs from wikipedia for each kw
    # 6. Chunk each document
    # 7. Store chunks in store
    # 8. Similarity Search/context retrieval
    # 9. Limit context length
    # 10. Create answering prompt
    # 11. Generate answer using llm
    # 12. Extract choice from model response
    # 13. Save responses to txt file 
    # 14. Calculate accuracy

    # Load dataset
    logging.info('Loading dataset...')
    df = pd.read_csv(args.dataset)
    if args.num_samples != -1:
        df = df.head(args.num_samples)
    logging.info(f'Loaded {len(df)} questions')
    if args.verbose:
        df.info()
    print()


    # Create kw prompt for each question
    logging.info('Creating keyword prompts from questions...')
    kw_prompts = []
    for _, row in df.iterrows():
        question = row['prompt']
        choices : List[str] = [row['A'], row['B'], row['C'], row['D'], row['E']]
        if args.kw_from_choices:
            kw_prompts.append(get_keyword_generation_prompt(question, choices))
        else:
            kw_prompts.append(get_keyword_generation_prompt(question))
    logging.info(f'Successfully created {len(kw_prompts)} keyword prompts')
    logging.debug(f'Keyword prompt 1: {kw_prompts[0]}')
    print()

    # Genrate keywords for each question 
    keywords = []
    if args.load_kw:
        logging.info(f'Loading keywords from file...')
        with open("data/keywords/keywords_train.txt", "r", encoding="utf-8") as f:
            for line in f:
                keywords.append(line.strip().split(","))
        logging.info(f'Successfully loaded keywords for {len(keywords)} questions')
        if args.verbose:
            print(keywords)
    else:
        logging.info(f'Generating keywords using "{args.kw_model}"...')
        for i, kw_prompt in enumerate(kw_prompts):
            keywords.append(generate_keywords(args.kw_model, kw_prompt))
            print(f'Keywords for question {i}: {", ".join(keywords[i])}')
        logging.info(f'Successfully generated keywords for {len(keywords)} questions')
    print()

    # save keywords for fast debugging
    makedirs('data/keywords', exist_ok=True)
    with open(f'data/keywords/keywords_train.txt', 'w', encoding='utf-8') as f:
        for keyword_list in keywords:
            f.write(f'{", ".join(keyword_list)}\n')
    print()

    # Download docs from wikipedia for each kw
    from src.web import download_web_pages_by_keywords
    logging.info("Downloading docs for each kw from wikipedia...")
    for keyword_list in keywords:
        download_web_pages_by_keywords(keywords=keyword_list, out_dir=f'search_results')
    logging.info(f'Successfully downloaded docs for {len(keywords)} questions')
    print()
    return

    # 6. Chunk each document
    from .rag import chunk_document
    chunks = []
    for doc in docs:
        chunks.append(chunk_document(doc, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap))

    print(chunks)

    # 7. Store chunks in store
    from .rag import store_chunks_in_db
    store_chunks_in_db(chunks)

    # 8. Similarity Search/context retrieval
    from .rag import retrieve_relevant_chunks
    relevant_chunks = []
    for chunk in chunks:
        relevant_chunks.append(retrieve_relevant_chunks(chunk))

    print(relevant_chunks)

    # 9. Limit context length

    # 10. Create answering prompt
    from .prompts import get_answer_prompt
    answer_prompts = []
    for _, row in df.iterrows():
        answer_prompts.append(get_answer_prompt(row['question'], row['context']))

    print(answer_prompts)

    # 11. Generate answer using llm
    from .llm import generate_response
    answer_responses = []
    for prompt in answer_prompts:
        answer_responses.append(generate_response(prompt, args.answer_model))

    print(answer_responses)

    # 12. Extract choice from model response
    from .prompts import get_choices
    choices = []
    for answer_response in answer_responses:
        choices.append(get_choices(answer_response))

    print(choices)

    # 13. Save responses to txt file
    import os
    os.makedirs('data/rag', exist_ok=True)
    for i, (question, answer, choices) in enumerate(zip(df['question'], df['answer'], choices)):
        with open(f'data/rag/rag_{i}.txt', 'w') as f:
            f.write(f'Question: {question}\n')
            f.write(f'Answer: {answer}\n')
            f.write(f'Choices: {choices}\n')

    # 14. Calculate accuracy
    from .prompts import get_accuracy
    accuracies = []
    for i, (question, answer, choices) in enumerate(zip(df['question'], df['answer'], choices)):
        accuracies.append(get_accuracy(answer, choices))
    
    print(accuracies)
    print(f'Average accuracy: {sum(accuracies) / len(accuracies)}')


if __name__ == "__main__":
    main()