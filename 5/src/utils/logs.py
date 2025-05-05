from model.document import Document


def write_vocabulary(documents, file):
    vocab = Document.build_vocabulary(documents)
    print("vocabulary count:", len(vocab))
    with open(file, "w", encoding="utf-8") as f:
        for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{key} {value}\n")


def write_index(index, file):
    with open(file, "w", encoding="utf-8") as f:
        f.write(str(index))


def print_top_k(iterable: list, k: int):
    width = len(str(k))
    for i, item in enumerate(iterable, 1):
        print(f"{str(i).rjust(width)}. {item}")
        if i >= k:
            break
