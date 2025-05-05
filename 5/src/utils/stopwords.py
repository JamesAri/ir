def load_stopwords(file_path):
    with open(file_path, encoding="utf-8") as f:
        return set(f.read().splitlines())
