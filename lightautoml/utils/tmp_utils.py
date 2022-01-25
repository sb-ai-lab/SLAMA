import pickle
from traceback import extract_stack
from typing import Any


def log_data(name: str, data: Any) -> None:
    log_record = {"stacktrace": extract_stack(), "data": data}
    with open(f"datalog_{name}.pickle", "wb") as f:
        pickle.dump(log_record, f)


def read_data(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data