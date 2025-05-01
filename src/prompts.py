from typing import List

def get_search_prompt(question: str) -> str:
    '''Generates a prompt for the keyword generation model. The output of the model when passed this prompt should hopefully be a comma separated list of keywords.

    Args:
        question (str): The question to answer.

    Returns:
        str: The prompt for the keyword generation model.
    '''

    return f"""you are a scientist who is trying to answer a question about a scientific topic. i can provide you with a bunch of text from wikipedia articles, but you need to give me keywords to search for to find the answer.

here is the question: {question}

what keywords do you need me to search for you to answer the question?
i can only search for up to 3 keywords, so be picky and choose the most relevant keywords.

IMPORTANT: DO NOT EXPLAIN OR CHAT, ONLY RESPOND WITH THE KEYWORDS(COMMA SEPERATED)
comma separated list of keywords:
"""

def get_retrieval_prompt(question: str, choices: List[str]) -> str:
    '''Generates a prompt for the retrieval model. This prompt is passed to the embedding model and then is compared with chunks in db.

    Args:
        question (str): The question to answer.
        choices (List[str]): The possible choices for the question.

    Returns:
        str: The prompt for the retrieval model.
    '''

    if len(choices) != 5:
        raise ValueError("choices must be of length 5")

    return f"""Answer this multiple choice question that is about a scientific topic.

Question: {question}

Choices:
Choice A) {choices[0]}
Choice B) {choices[1]}
Choice C) {choices[2]}
Choice D) {choices[3]}
Choice E) {choices[4]}

Answer:
"""

def get_answer_prompt(question: str, choices: List[str], context: str) -> str:
    '''Generates a prompt for the answering model.
    
    Args:
        question (str): The question to answer.
        choices (List[str]): The possible choices for the question.
        context (str): The context from which to generate the answer.
    
    Returns:
        str: The prompt for the answering model.
    '''

    if len(choices) != 5:
        raise ValueError("choices must be of length 5")

    return f"""Given the following question, context, and choices, carefully analyze each choice to determine which one is correct. Break down the problem step by step:

Context: {context}

Question: {question}

Choices:
Choice A) {choices[0]}
Choice B) {choices[1]}
Choice C) {choices[2]}
Choice D) {choices[3]}
Choice E) {choices[4]}

Very important notes:
- Start by reading the question carefully. analyze what the question is asking for yourself before going through the choices.
- Carefully consider the relationship or concept described in the question and context.
- Evaluate each choice against the context and the question, one by one.
- Check if each choice is correct, partially correct, or incorrect based on the context and the information provided.
- Provide a final answer after considering all choices.

It's ok to talk and think about the problem but the last line of your answer should be the string 'Answer:' followed by either A, B, C, D, or E.
"""

def get_answer_prompt_old(question: str, choices: List[str], context: str) -> str:
    '''Generates a prompt for the answering model.
    
    Args:
        question (str): The question to answer.
        choices (List[str]): The possible choices for the question.
        context (str): The context from which to generate the answer.
    
    Returns:
        str: The prompt for the answering model.
    '''

    if len(choices) != 5:
        raise ValueError("choices must be of length 5")

    return f"""Answer this multiple choice question using the context below.
You should be very careful to answer the question correctly. So think about the problem step by step.

Context: {context}

Question: {question}

Choices:
Choice A) {choices[0]}
Choice B) {choices[1]}
Choice C) {choices[2]}
Choice D) {choices[3]}
Choice E) {choices[4]}

It's ok to talk and think about the problem but the last line of your answer should be the string 'Answer:' followed by either A, B, C, D, or E.
Example answer:
'Based on the context, the best answer would be Oxygen and because Choice A) Oxygen,
Answer: A'
"""