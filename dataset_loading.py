import os
import json
from typing import List, Tuple, Dict, Any
# import kagglehub

# def load_data_from_kagglehub() -> Tuple[List[str], List[str], List[Dict[str, Any]]]:

#     # Download dataset
#     path = kagglehub.dataset_download("elvinagammed/chatbots-intent-recognition-dataset")
#     # print("Dataset:", path)

#     # files are inside the path
#     print("Files in dataset directory:", os.listdir(path))

#     # storing the path
#     json_path = os.path.join(path, "Intent.json")  # ← Use "Intent.json", not "intents.json"

#     # Loading the JSON file
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     # Extracting sentences and labels
#     texts, labels = [], []
#     for intent in data["intents"]:
#         for sentence in intent["text"]:
#             texts.append(sentence)
#             labels.append(intent["intent"])

#     return texts, labels, data["intents"]


def load_data_from_local(path: str = "data/Intent.json") -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for sentence in intent["text"]:
            texts.append(sentence)
            labels.append(intent["intent"])

    return texts, labels, data["intents"]

texts, labels, intents_data = load_data_from_local()

print("\nSample training data:")
for i in range(3):
    print(f"  {texts[i]} → {labels[i]}")

print("\nTotal samples:", len(texts))