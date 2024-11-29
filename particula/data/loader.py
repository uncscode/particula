import warnings

def filter_list(data, character):
    filtered_data = [item for item in data if character in item]
    if len(filtered_data) / len(data) > 0.5:
        warnings.warn(
            f"More than 0.5 of the rows have been filtered out based on the character: {character}.",
            UserWarning
        )
    return filtered_data

def test_filter_list():
    data = ["apple", "banana", "cherry", "date", "elderberry"]
    character = "a"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        filtered_data = filter_list(data, character)
    assert len(filtered_data) == 3

if __name__ == "__main__":
    test_filter_list()
