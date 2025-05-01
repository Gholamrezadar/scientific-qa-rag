from typing import List, Tuple


def evaluate_answer_list(answer_list: List[str], correct_answers: List[str]) -> float:
    '''Evaluates the accuracy of the answer list.

    Args:
        answer_list (List[str]): The list of answers.
        correct_answers (List[str]): The list of correct answers.

    Returns:
        float: The accuracy of the answer list.
    '''
    if len(answer_list) > len(correct_answers):
        raise ValueError("answer_list is longer than correct_answers")

    correct = 0
    for i in range(len(answer_list)):
        if answer_list[i] == correct_answers[i]:
            correct += 1

    return correct / len(answer_list)


def evaluate_ollama_model(model_name: str, dataset_file: str, num_questions: int = -1) -> Tuple[List[str], List[str], float]:
    '''Evaluates the accuracy of the answer list.

    Args:
        model_name (str): The name of the model to use.
        dataset_file (str): The file containing the dataset.
        num_questions (int, optional): The number of questions to use. Defaults to -1, which means all questions.

    Returns:
        List[str]: The ABCDE answer_list.
        List[str]: The raw model_responses.
        float: The accuracy of the answer list.
    '''

    from src.llm import get_answers_by_model
    from src.utils import extract_answered_choice
    from src.eval import evaluate_answer_list
    from src.utils import get_correct_answers

    correct_answers = get_correct_answers(dataset_file)
    model_responses = get_answers_by_model(model_name, dataset_file, num_questions)
    answer_list = [extract_answered_choice(response) for response in model_responses]
    accuracy = evaluate_answer_list(answer_list, correct_answers)

    return answer_list, model_responses, accuracy