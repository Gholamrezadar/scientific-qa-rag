from typing import List
from tqdm.autonotebook import tqdm


def get_answers_by_model(model_name: str, dataset_file: str, num_questions: int = -1) -> List[str]:
    '''Applies the answer_prompt to all questions in the dataset and returns the answer_list.
    
    Args:
        model_name (str): The name of the model to use.
        dataset_file (str): The file containing the dataset.
        num_questions (int, optional): The number of questions to use. Defaults to -1, which means all questions.

    Returns:
        List[str]: The answer_list.
    '''

    import pandas as pd
    from src.prompts import get_answer_prompt
    from langchain_ollama import ChatOllama
    from langchain.schema import HumanMessage

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