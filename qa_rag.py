import argparse
import logging
import sys
from os import makedirs

from typing import List

import pandas as pd

print("Initializing Langchain...")
from src.prompts import get_keyword_generation_prompt, get_answer_prompt
from src.llm import generate_keywords 
from src.web import download_web_pages_by_keywords
from src.rag import load_documents, chunk_document, store_chunks_in_db, retrieve_relevant_chunks
from src.llm import generate_answers
from src.utils import extract_choice_from_response
from src.eval import evaluate_answer_list


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
    parser.add_argument("--max-context", type=int, default=1000, help="Maximum context length per question. set based on your answer generation model's context length.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    
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
                if line.strip() != '':
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

    # Download docs from wikipedia for each kw
    logging.info("Downloading docs for each kw from wikipedia...")
    for keyword_list in keywords:
        download_web_pages_by_keywords(keywords=keyword_list, out_dir=f'search_results')
    logging.info(f'Successfully downloaded docs for {len(keywords)} questions')
    print()

    # Chunk docs and store them in vector db 
    logging.info("Chunking docs and storing them in vector db...")
    retrieved_docs = load_documents(doc_dir='search_results')
    chunks = chunk_document(document=retrieved_docs[0], chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    store_chunks_in_db(chunks=chunks)
    logging.info(f'Successfully chunked docs and stored them in vector db for')
    print()

    # Retrieve relevant context for each question
    logging.info(f"Retrieving relevant context for each question(total: {len(df)})...")
    contexts = []
    for _, row in df.iterrows():
        contexts.append(retrieve_relevant_chunks(question=row['prompt']))
    logging.info(f'Successfully retrieved relevant context for {len(contexts)} questions')
    print()

    # Limit context length
    logging.info(f"Limiting context length to {args.max_context} characters...")
    contexts = [context[:args.max_context] for context in contexts]
    logging.info(f'Successfully limited context length for {len(contexts)} questions')
    print()

    # Create answering prompts
    logging.info("Creating answering prompts...")
    answer_prompts = []
    for i, row in df.iterrows():
        question = row['prompt']
        context = contexts[i]
        choices = [row['A'], row['B'], row['C'], row['D'], row['E']]
        answer_prompts.append(get_answer_prompt(question, choices, context))
    logging.info(f'Successfully created {len(answer_prompts)} answering prompts')

    # Save answer prompts to file for review
    with open(f'data/prompts/{args.answer_model.replace(":","_")}_prompts.txt', 'w', encoding='utf-8') as f:
        for prompt in answer_prompts:
            f.write(prompt)
            f.write('\n')
            f.write("- "*40)
            f.write('\n')
    logging.info(f'Saved answer prompts to {f"data/prompts/{args.answer_model.replace(":","_")}_prompts.txt"}')
    print()


    # Generate responses using llm
    logging.info(f"Generating answers for {len(answer_prompts)} prompts...")
    answer_responses = generate_answers(args.answer_model, answer_prompts)
    logging.info(f'Successfully generated {len(answer_responses)} answers')
    # Save raw responses to txt file for review
    with open(f'data/responses/{args.answer_model.replace(":","_")}_responses.txt', 'w', encoding='utf-8') as f:
        for response in answer_responses:
            f.write(response)
            f.write('\n')
            f.write("- "*40)
            f.write('\n')
    logging.info(f'Saved responses to {f"data/responses/{args.answer_model.replace(":", "_")}_responses.txt"}')
    print()

    # Read correct choices from dataset if available
    ground_truth_choices = []
    if 'answer' in df.columns:
        ground_truth_choices = df['answer'].tolist()

    # Validate gt
    for choice in ground_truth_choices:
        if choice not in "ABCDE":
            raise ValueError(f"Invalid choice in dataset: {choice}")
    
    # Extract choices from raw responses
    logging.info("Extracting choices from raw responses...")
    choices = []
    for answer_response in answer_responses:
        choices.append(extract_choice_from_response(answer_response))
    print(f"Y_pred: {choices}")
    if len(choices) == len(ground_truth_choices):
        print(f"Y_true: {ground_truth_choices}")

    logging.info(f'Successfully extracted choices for {len(choices)} responses')

    # Save responses to txt file
    with open(f"{args.answer_model.replace(":","_")}_ypred.txt", 'w', encoding="utf8") as f:
        for choice in choices:
            f.write(choice + ', ')
    logging.info(f"Wrote prediction choices to {f'{args.answer_model.replace(":","_")}_ypred.txt'}")
    print()

    # Calculate accuracy
    if len(choices) == len(ground_truth_choices):
        accuracy = evaluate_answer_list(choices, ground_truth_choices)
        print(f"Accuracy for {args.answer_model}: {accuracy}")
    print()

if __name__ == "__main__":
    main()