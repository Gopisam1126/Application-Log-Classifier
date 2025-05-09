from regex_processor import classify_with_regex
from bert_processor import bert_classify
from llm_proecssor import classify_with_llm

def process_entries(entries):
    """
    entries: list of tuples (origin, message)
    Returns a list of classification labels.
    """
    results = []
    for origin, message in entries:
        result = process_single(origin, message)
        results.append(result)
    return results

def process_single(origin, message):
    """
    origin: string indicating the source system
    message: log text to classify
    """
    if origin == "LegacyCRM":
        return classify_with_llm(message)
    else:
        label = classify_with_regex(message)
        if label == "Other":
            label = bert_classify(message)
        return label

def process_file(input_path):
    """
    input_path: path to a CSV file with 'source' and 'log_message' columns
    Reads, classifies, and writes out a new CSV.
    """
    import pandas as pd

    data_frame = pd.read_csv(input_path)
    if "source" not in data_frame.columns or "log_message" not in data_frame.columns:
        raise ValueError("Input CSV must have 'source' and 'log_message' columns.")
    
    # Apply classification
    tuples = list(zip(data_frame["source"], data_frame["log_message"]))
    data_frame["predicted_label"] = process_entries(tuples)

    output_path = "resources/processed_output.csv"
    data_frame.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    output = process_file("test.csv")
    print(f"Processed file saved to: {output}")
