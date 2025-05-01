from src.eval import evaluate_ollama_model

print("Evaluating Gemma3:1b on the whole dataset")
answer_list, model_responses, accuracy = evaluate_ollama_model(model_name="gemma3:1b", dataset_file="data/train_data.csv", num_questions=-1)

print(f"answer_list: {answer_list}")

for i, response in enumerate(model_responses):
    print(f"{i+1}/50:")
    print(response)
    print("- " * 20)

print(f"accuracy: {accuracy}")