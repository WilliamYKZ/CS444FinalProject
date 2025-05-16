import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def test_reward_model(model_name_or_path, test_texts=None):
    """
    Attempt to load `model_name_or_path` into a `text-classification` pipeline
    and see if it returns a numeric 'score' we can interpret as a reward.

    Returns:
        bool: True if it can be used (at least from a shape/output standpoint),
              False otherwise.
    """

    if test_texts is None:
        test_texts = [
            "Question: How can I sort a list in Python?\n\nAnswer: You can use the built-in sort method.",
            "Question: What is the capital of France?\n\nAnswer: The capital of France is Paris."
        ]

    # Try loading into a text-classification pipeline
    try:
        # Important: using `return_all_scores=True` to see whether we get multiple labels & scores
        classifier = pipeline(
            "text-classification",
            model=model_name_or_path,
            return_all_scores=True
        )
    except Exception as e:
        print(f"Could not load model into a text-classification pipeline. Error: {e}")
        return False

    # Test the pipeline on some sample texts
    for text in test_texts:
        try:
            results = classifier(text)

            # `results` is typically a list (batch dimension),
            # e.g. [ [ {'label': 'LABEL_0', 'score': 0.8}, ... ] ]
            if not isinstance(results, list) or len(results) == 0:
                print("Pipeline output shape/format not as expected.")
                return False

            # We typically get one list per input item. Let's just look at the first one
            first_item = results[0]
            if not isinstance(first_item, list) or len(first_item) == 0:
                print("Missing classification results inside pipeline output.")
                return False

            # Check if we have the 'score' field in the dictionary
            # e.g. {'label': 'POSITIVE', 'score': 0.93}
            if not all("score" in d for d in first_item):
                print("No 'score' field found in pipeline output.")
                return False

            # If everything checks out, we can interpret 'score' as a numeric value
            # Possibly reduce them to a single scalar, if you want
            # (the code in your PPO example just took the first dictionary's 'score').
            print(f"Sample text: {text}")
            print(f"Pipeline output: {results}\n")

        except Exception as e:
            print(f"Error during inference on text '{text}': {e}")
            return False

    # If we made it here, the pipeline can run, and we do get numeric scores
    print("Model outputs at least one score and can be used as a reward model (mechanically).")
    return True


if __name__ == "__main__":
    model_name = "/home/exouser/Desktop/DeepSeek-R1-Distill-Qwen-7B"  # Replace with your model

    can_use = test_reward_model(model_name)
    if can_use:
        print("Yes, this model can be used as a reward model (output format is valid).")
    else:
        print("No, the model's output format isn't appropriate for a reward model.")
