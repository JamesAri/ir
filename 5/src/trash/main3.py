import json
import pickle
from tqdm import tqdm
from engines.boolean.engine import BooleanSearchEngine
from model.document import Document
from src.model.positional_index import PositionalIndex


def parse_courseware_cz_demo_document_text(doc):
    doc_id = f"(Doc Id: {doc["id"]})" or "(Has no Document Id)"
    title = doc["title"] or ""
    text = doc["text"] or ""
    return " ".join([doc_id, title, text])


def save_index(index, file):
    with open(file, "wb") as f:
        pickle.dump(index, f)


def load_index(file):
    with open(file, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("\nRunning Boolean search task on cz data from courseware (documents.json)")

    pipeline = pre.PreprocessingPipeline(
        [
            pre.LowercasePreprocessor(),
        ]
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "out"))
    input_file = os.path.join(script_dir, "..", "data", "documents.json")
    index_file = os.path.join(out_dir, "index.pkl")
    os.makedirs(out_dir, exist_ok=True)
    # NOTE:
    # boolean_queries_standard_10.txt and boolean_queries_standard_100.txt not parsable
    # with the provided parser.
    query_file = os.path.join(script_dir, "..", "data", "boolean_queries_simple.txt")

    if os.path.exists(index_file):
        print(f"Loading inverted index from file {index_file}")
        index: PositionalIndex = load_index(index_file)
        print(f"Loaded inverted index with {index.get_documents_count()} documents")
    else:
        documents: list[Document] = []
        with open(input_file, "r", encoding="utf-8") as f:
            tqdm_json = tqdm(json.load(f), desc="Tokenizing documents")
            for doc in tqdm_json:
                documents.append(
                    Document(parse_courseware_cz_demo_document_text(doc)).tokenize()
                )

        tqdm_documents = tqdm(documents)
        for i, doc in enumerate(tqdm_documents, 1):
            tqdm_documents.set_description(f"Processing document {i}/{len(documents)}")
            doc.preprocess(pipeline)

        print("Creating inverted index for preprocessed documents")
        index = PositionalIndex(documents, show_progress=True)

        print("Saving inverted index to file (pickle)")
        print("Note: should be around 1,1GB for the courseware data")
        save_index(index, index_file)

    search_engine = BooleanSearchEngine(index=index)

    with open(query_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f.readlines()]

    K = 3
    MAX_OUTPUT_LEN = 50

    for query in queries:
        print(f"\nQUERY['{query}']:")
        documents = search_engine.search(query)
        for i, doc in enumerate(documents, 1):
            print(f"{i}. {doc[:MAX_OUTPUT_LEN]}")
            if i >= K:
                print(f"Skipped {len(documents) - K} documents")
                break

    print("\n=== Finished processing all queries ===")
    print(f"Showing only {K} documents per query")
    print(f"Showing only first {MAX_OUTPUT_LEN} characters of each document")
