from typing import List

# TODO: rename to extract_choice_from_answer()
def extract_answered_choice(model_response: str) -> int:
    '''Extracts the answered choice from the model response.

    Args:
        model_response (str): The response from the model.

    Returns:
        int: The answered choice. 0:A, 1:B, 2:C, 3:D, 4:E
    
    Raises:
        ValueError: If the model response does not start with 'Answer:' or if the model response does not include A, B, C, D, or E at the end.
    '''

    last_line = model_response.split('\n')[-1]

    if last_line.startswith('Answer:'):
        last_line = last_line[len('Answer:'):].strip()
    else:
        print("bad response!")
        return 'A'
        # raise ValueError("Model response does not start with 'Answer:'")

    if last_line.startswith('A'):
        return 'A'
    elif last_line.startswith('B'):
        return 'B'
    elif last_line.startswith('C'):
        return 'C'
    elif last_line.startswith('D'):
        return 'D'
    elif last_line.startswith('E'):
        return 'E' 
    else:
        print("bad response!")
        return 'A'
        # raise ValueError("Model response does not include A, B, C, D, or E at the end.")

def save_prompts_to_file(dataset_file: str, result_file:str) -> None:
    '''Applies the answer_prompt to all questions in the dataset and saves the prompts to a file. This will be used to asses GUI based models like ChatGPT.'''

    import pandas as pd
    from src.prompts import get_answer_prompt

    df = pd.read_csv(dataset_file)
    prompts = []
    correct_answers = []

    for i, row in df.iterrows():
        question = row['prompt']
        choices = [row['A'], row['B'], row['C'], row['D'], row['E']]
        context = ""
        answer = row['answer']
        prompt = get_answer_prompt(question, choices, context)
        prompts.append(prompt)
        correct_answers.append(answer)

    with open(result_file, 'w', encoding="utf8") as f:
        f.write('\n\n\n\n'.join(prompts))
    
    with open(result_file.replace('.txt', '_correct_answers.txt'), 'w', encoding="utf8") as f:
        f.write(', '.join(correct_answers))

def get_correct_answers(dataset_file: str) -> List[str]:
    '''Gets the correct answers from the dataset file.

    Args:
        dataset_file (str): The file containing the dataset.

    Returns:
        List[str]: The correct answers.
    '''

    import pandas as pd

    df = pd.read_csv(dataset_file)
    correct_answers = []

    for i, row in df.iterrows():
        correct_answers.append(row['answer'])

    return correct_answers