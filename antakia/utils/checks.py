def is_valid_model(model) -> bool:
    return callable(getattr(model, "score")) and callable(getattr(model, "predict"))
