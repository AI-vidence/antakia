def is_valid_model(model) -> bool:
    """
    checks whether the customer model is a valid model
    """
    return callable(getattr(model, "score")) and callable(getattr(model, "predict"))
