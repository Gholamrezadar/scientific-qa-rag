import argparse
import logging

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=level
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Answer Multiple Choice Scientific Questions using RAG and Internet Search", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--kw-model", default='gemma3:1b', help="Model used for keyword extraction")
    parser.add_argument("--answer-model", default='gemma3:1b', help="Model used to generate answers")
    parser.add_argument("--embed-model", default='nomic-embed-text', help="Model used for embedding")
    parser.add_argument("--kw-from-choices", action="store_true", help="Extract keywords from the choices too (if false, only extract from the question)")
    # parser.add_argument("--wiki-max-length", type=int, default=1000, help="Max length of fetched Wikipedia content")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of chunks for retrieval")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    logging.info("Loading dataset...")
    dataset = [1, 2, 3]

    for idx in dataset:
        question = f"Fake question {idx}"
        choices = {"A": "Fake choice A", "B": "Fake choice B", "C": "Fake choice C", "D": "Fake choice D", "E": "Fake choice E"}

        logging.debug(f"Question: {question}")
        logging.debug(f"Choices: {choices}")

        logging.info("Generating keywords...")
        keywords = ["Fake keyword 1", "Fake keyword 2", "Fake keyword 3"]
        logging.debug(f"Generated keywords: {keywords}")

        logging.info("Downloading documents from the web...")
        documents = ["Fake document 1", "Fake document 2", "Fake document 3"]
        logging.info(f"Fetched {len(documents)} documents")

        logging.info("Chunking and embedding documents...")
        print()

    
    logging.info("Evaluating the model...")
    accuracy = 0.73
    logging.info(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()