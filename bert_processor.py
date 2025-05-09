from sentence_transformers import SentenceTransformer
import joblib

embedder = SentenceTransformer("all-MiniLM-L6-v2")
classifier_model = joblib.load("model/log_model.joblib")

def bert_classify(text):
    """
    Generate a label for a log message using a BERT-based classifier.
    Returns 'Unclassified' if confidence is below threshold.
    """
    vectorized = embedder.encode([text])
    probs = classifier_model.predict_proba(vectorized)[0]
    if max(probs) < 0.5:
        return "Unclassified"
    prediction = classifier_model.predict(vectorized)[0]
    return prediction

if __name__ == "__main__":
    sample_logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for entry in sample_logs:
        lbl = bert_classify(entry)
        print(f"{entry} -> {lbl}")
