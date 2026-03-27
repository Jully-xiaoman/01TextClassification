from transformers import BertForSequenceClassification

def create_model(config):
    model = BertForSequenceClassification.from_pretrained(
        config["model_path"],
        num_labels=config["num_labels"]
    )
    return model