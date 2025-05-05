import os
import json
import pickle
from tqdm import tqdm

from model import Document
from src.model.positional_index import PositionalIndex
from engines.tf_idf_engine import TfIdfSearchEngine
from utils.stopwords import load_stopwords


def save_index(index, file):
    with open(file, "wb") as f:
        pickle.dump(index, f)


def load_index(file):
    with open(file, "rb") as f:
        return pickle.load(f)


if __name__ == "X__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "documents.json")
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "out"))
    os.makedirs(out_dir, exist_ok=True)

    index_file = os.path.join(out_dir, "index.pkl")

    if os.path.exists(index_file):
        print(f"Loading positional index from file {index_file}")
        index: PositionalIndex = load_index(index_file)
        print(f"Loaded positional index with {index.get_documents_count()} documents")
    else:

        documents = []
        with open(input_file, "r", encoding="utf-8") as f:
            tqdm_json = tqdm(json.load(f), desc="Tokenizing documents")
            for doc in tqdm_json:
                documents.append(
                    Document(parse_courseware_cz_demo_document_text(doc)).tokenize()
                )

        ###### Without Preprocessing ######

        vocab_1_file = os.path.join(out_dir, "vocab1.txt")
        write_vocabulary(documents, vocab_1_file)

        ###### Preprocessing ######

        tqdm_documents = tqdm(documents, desc="Preprocessing documents")
        for doc in tqdm_documents:
            doc.preprocess(pipeline)

        vocab_2_file = os.path.join(out_dir, "vocab2.txt")
        write_vocabulary(documents, vocab_2_file)

        ###### Positional Index ######

        print("Creating positional index for preprocessed documents")
        index = PositionalIndex(documents, show_progress=True)

        print("Saving positional index to file (pickle)")
        print("Note: should be around 1,1GB for the courseware data")
        save_index(index, index_file)

        ###### Output ######
        print(f"Vocabulary written to {vocab_1_file} (without preprocessing)")
        print(f"Vocabulary written to {vocab_2_file} (preprocessed)")
        print(f"Index saved to {index_file} (raw)")

    search = TfIdfSearchEngine(index, pipeline)
    k = 10
    result = search.search(query="Klaus varoval", k=k, method="ltu.ltc")
    print_top_k(result, k=k)

if __name__ == "__main__":
    print("\nRunning 1. Task from Courseware")
    documents = [
        Document("Plzeň je krásné město a je to krásné místo").tokenize(),
        Document("Ostrava je ošklivé místo").tokenize(),
        Document("Praha je také krásné město Plzeň je hezčí").tokenize(),
    ]

    search = TfIdfSearchEngine(PositionalIndex(documents))
    k = 10
    result = search.search(query="krásné město", k=k, method="ltc.ltc")
    print_top_k(result, k)


if __name__ == "X__main__":
    print("\nRunning 2. Task from Courseware")
    documents = [
        Document("tropical fish include fish found in tropical enviroments").tokenize(),
        Document("fish live in a sea").tokenize(),
        Document("tropical fish are popular aquarium fish").tokenize(),
        Document("fish also live in Czechia").tokenize(),
        Document("Czechia is a country").tokenize(),
    ]

    search = TfIdfSearchEngine(PositionalIndex(documents))
    k = 10

    query = "tropical fish sea"
    print(f"\nQuery: {query}")
    result = search.search(query=query, k=k, method="ltc.ltc")
    print_top_k(result, k)

    query = "tropical fish"
    print(f"\nQuery: {query}")
    result = search.search(query=query, k=k, method="ltc.ltc")
    print_top_k(result, k)


if __name__ == "X__main__":
    print("\nRunning 5. Task from Courseware - searching on our own data")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "sample.json")
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "out"))
    os.makedirs(out_dir, exist_ok=True)

    documents = []
    with open(input_file, "r", encoding="utf-8") as f:
        tqdm_json = tqdm(json.load(f), desc="Tokenizing documents")
        for doc in tqdm_json:
            documents.append(
                Document(parse_zatrolene_hry_document_text(doc)).tokenize()
            )

    ###### Without Preprocessing ######

    vocab_1_file = os.path.join(out_dir, "vocab1.txt")
    write_vocabulary(documents, vocab_1_file)

    index_1_file = os.path.join(out_dir, "index1.json")
    write_index(PositionalIndex(documents), index_1_file)

    ###### Preprocessing ######

    tqdm_documents = tqdm(documents)
    for i, doc in enumerate(tqdm_documents, 1):
        tqdm_documents.set_description(f"Processing document {i}/{len(documents)}")
        doc.preprocess(pipeline)

    vocab_2_file = os.path.join(out_dir, "vocab2.txt")
    write_vocabulary(documents, vocab_2_file)

    index_2_file = os.path.join(out_dir, "index2.json")
    write_index(PositionalIndex(documents), index_2_file)

    ###### SearchEngine ######

    search = TfIdfSearchEngine(PositionalIndex(documents), pipeline)

    k = 10
    result = search.search(
        query="Root Hra pouze jednou hraná, super stav", k=k, method="ltu.ltc"
    )
    print_top_k(result, k=k)
    result = search.search(query="War nehrano", k=k, method="ltu.ltc")
    print_top_k(result, k=k)

    ###### Output ######
    print(f"Vocabulary written to {vocab_1_file} (without preprocessing)")
    print(f"Vocabulary written to {vocab_2_file} (preprocessed)")
    print(f"Positional index written to {index_1_file} (without preprocessing)")
    print(f"Positional index written to {index_2_file} (preprocessed)")
