def extract_answered_choice(model_response: str) -> int:
    '''Extracts the answered choice from the model response.

    Args:
        model_response (str): The response from the model.

    Returns:
        int: The answered choice. 0:A, 1:B, 2:C, 3:D, 4:E
    
    Raises:
        ValueError: If the model response does not start with 'Answer:' or if the model response does not include A, B, C, D, or E at the end.
    '''

    if model_response.startswith('Answer:'):
        model_response = model_response[len('Answer:'):].strip()
    else:
        raise ValueError("Model response does not start with 'Answer:'")

    if model_response.startswith('A'):
        return 0
    elif model_response.startswith('B'):
        return 1
    elif model_response.startswith('C'):
        return 2
    elif model_response.startswith('D'):
        return 3
    elif model_response.startswith('E'):
        return 4
    else:
        raise ValueError("Model response does not include A, B, C, D, or E at the end.")

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

