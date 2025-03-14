import os
import json
from tqdm import tqdm

from document import Document
from tokenizer import TokenType
import preprocess as pre
from utils import load_stopwords
from positional_index import PositionalIndex
from search_engine import SearchEngine

cs_stop_words = load_stopwords(os.path.join(os.path.dirname(__file__), "..", "stopwords", "stopwords-cs.txt"))

pipeline = pre.PreprocessingPipeline([
    pre.TokenFilterPreprocessor([TokenType.URL, TokenType.PUNCT]),
    pre.StopWordsPreprocessor(cs_stop_words),
    pre.LowercasePreprocessor(),
    pre.HtmlStripPreprocessor(),
    pre.WhitespaceStripPreprocessor(),
    pre.NumberPreprocessor(),
    pre.TokenLengthPreprocessor(min_length=2),
    # pre.LemmatizationPreprocessor(), // tunred off so you can test search engine faster
    pre.UnidecodePreprocessor(),
])

def parse_zatrolene_hry_document_text(doc):
    return " ".join([doc["Prodavane_predmety"], doc["Popisek"]])


def parse_courseware_cz_demo_document_text(doc):
    return " ".join([doc["title"] or "", doc["text"] or ""])


def write_vocabulary(documents, file):
    vocab = Document.build_vocabulary(documents)
    print('vocabulary count:', len(vocab))
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

if __name__ == '__main__':
    print('\nRunning 1. Task from Courseware')
    documents = [
        Document('Plzeň je krásné město a je to krásné místo').tokenize(),
        Document('Ostrava je ošklivé místo').tokenize(),
        Document('Praha je také krásné město Plzeň je hezčí').tokenize(),
    ]

    search = SearchEngine(PositionalIndex(documents))
    k = 10
    result = search.search(query="krásné město", k=k, method='ltc.ltc')
    print_top_k(result, k)
    

if __name__ == '__main__':
    print('\nRunning 2. Task from Courseware')
    documents = [
        Document('tropical fish include fish found in tropical enviroments').tokenize(),
        Document('fish live in a sea').tokenize(),
        Document('tropical fish are popular aquarium fish').tokenize(),
        Document('fish also live in Czechia').tokenize(),
        Document('Czechia is a country').tokenize(),
    ]

    search = SearchEngine(PositionalIndex(documents))
    k = 10
    
    query = "tropical fish sea"
    print(f"\nQuery: {query}")    
    result = search.search(query=query, k=k, method='ltc.ltc')
    print_top_k(result, k)
    
    query = "tropical fish"
    print(f"\nQuery: {query}")
    result = search.search(query=query, k=k, method='ltc.ltc')
    print_top_k(result, k)


if __name__ == '__main__':
    print('\nRunning 5. Task from Courseware - searching on our own data')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "sample.json")
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "out"))
    os.makedirs(out_dir, exist_ok=True)

    documents = []
    with open(input_file, 'r', encoding="utf-8") as f:
        documents = [Document(parse_zatrolene_hry_document_text(doc)).tokenize() for doc in json.load(f)]

    ###### Without Preprocessing ######

    vocab_1_file = os.path.join(out_dir, "vocab1.txt")
    write_vocabulary(documents, vocab_1_file)

    index_1_file = os.path.join(out_dir, "index1.json")
    write_index(PositionalIndex(documents), index_1_file)

    ###### Preprocessing ######

    tqdm_documents = tqdm(documents)
    for i, doc in enumerate(tqdm_documents, 1):
        tqdm_documents.set_description(f"[>] Processing document {i}/{len(documents)}")
        doc.preprocess(pipeline)

    vocab_2_file = os.path.join(out_dir, "vocab2.txt")
    write_vocabulary(documents, vocab_2_file)

    index_2_file = os.path.join(out_dir, "index2.json")
    write_index(PositionalIndex(documents), index_2_file)

    ###### SearchEngine ######

    search = SearchEngine(PositionalIndex(documents), pipeline)

    k = 10
    result = search.search(query="Root Hra pouze jednou hraná, super stav", k=k, method='ltu.ltc')
    print_top_k(result, k=k)
    result = search.search(query="War nehrano", k=k, method='ltu.ltc')
    print_top_k(result, k=k)
    
    ###### Output ######
    print(f"Vocabulary written to {vocab_1_file} (without preprocessing)")
    print(f"Vocabulary written to {vocab_2_file} (preprocessed)")
    print(f"Positional index written to {index_1_file} (without preprocessing)")
    print(f"Positional index written to {index_2_file} (preprocessed)")
