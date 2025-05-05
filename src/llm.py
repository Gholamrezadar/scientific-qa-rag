from typing import List
from tqdm.autonotebook import tqdm
from prompts import get_keyword_generation_prompt, get_retrieval_prompt, get_answer_prompt
from utils import extract_keywords_from_answer
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import pandas as pd


def generate_keywords(keyword_prompt: str,model_name: str):
    '''Calls the LLM to generate keywords based on the keyword_prompt.

    Args:
        keyword_prompt (str): The prompt to use for keyword generation.
        model_name (str): The name of the model to use.

    Returns:
        List[str]: The keywords.
    '''

    chat_model = ChatOllama(model=model_name)

    full_prompt = get_keyword_generation_prompt(keyword_prompt)

    response = chat_model.invoke([HumanMessage(content=full_prompt)])

    # Extract the response content
    keywords = extract_keywords_from_answer(response.content)
    return keywords


# TODO: rename to generate_answers
def get_answers_by_model(model_name: str, dataset_file: str, num_questions: int = -1) -> List[str]:
    '''Applies the answer_prompt to all questions in the dataset and returns the answer_list.
    
    Args:
        model_name (str): The name of the model to use.
        dataset_file (str): The file containing the dataset.
        num_questions (int, optional): The number of questions to use. Defaults to -1, which means all questions.

    Returns:
        List[str]: The answer_list.
    '''

    chat_model = ChatOllama(model=model_name)

    df = pd.read_csv(dataset_file)

    # limit to num_questions
    if num_questions != -1 and num_questions < len(df):
        df = df.head(num_questions)

    prompts = []
    for i, row in df.iterrows():
        question = row['prompt']
        choices = [row['A'], row['B'], row['C'], row['D'], row['E']]
        context = ""
        prompt = get_answer_prompt(question, choices, context)
        prompts.append(prompt)

    answer_list = []
    for prompt in tqdm(prompts):
        response = chat_model.invoke([HumanMessage(content=prompt)])
        answer_list.append(response.content)
        # print(response.content.split('\n')[-1])

    return answer_list