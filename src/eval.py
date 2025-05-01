from typing import List


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
