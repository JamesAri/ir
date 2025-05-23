import json
import os
import pickle
import re

from tqdm import tqdm
from document import Document
from positional_index import PositionalIndex
import preprocess as pre

class Node:
    def evaluate(self, index: PositionalIndex, all_docs_ids: list[int]) -> set[int]:
        """Evaluate the node based on the given index and set of all documents."""
        raise NotImplementedError()


class TermNode(Node):
    def __init__(self, value: str):
        self.value = value.lower()  # Normalize to lowercase

    def __repr__(self):
        return f"Term({self.value})"

    def evaluate(self, index: PositionalIndex, all_docs_ids: list[int]) -> set[int]:
        """Retrieve document IDs from the inverted index."""
        postings = index.get_postings(self.value)
        return set(postings.keys()) if postings else set()


class NotNode(Node):
    def __init__(self, child):
        self.child = child

    def __repr__(self):
        return f"Not({self.child})"

    def evaluate(self, index: PositionalIndex, all_docs_ids: list[int]) -> set[int]:
        """Apply complement: NOT A → all_docs - A"""
        return all_docs_ids - self.child.evaluate(index, all_docs_ids)


class AndNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"And({self.left}, {self.right})"

    def evaluate(self, index: PositionalIndex, all_docs_ids: list[int]) -> set[int]:
        """Apply intersection: A AND B → A ∩ B"""
        return self.left.evaluate(index, all_docs_ids) & self.right.evaluate(index, all_docs_ids)


class OrNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Or({self.left}, {self.right})"

    def evaluate(self, index: PositionalIndex, all_docs_ids: list[int]) -> set[int]:
        """Apply union: A OR B → A u B"""
        return self.left.evaluate(index, all_docs_ids) | self.right.evaluate(index, all_docs_ids)


class BooleanParser:
    """
    Parses boolean expressions using the following grammar:
    expr: term (OR term)*
    term: factor (AND factor)*
    factor: [NOT] base
    base: LPAREN expr RPAREN | TERM
    """

    def __init__(self, text):
        self.tokens = self.tokenize(text)
        self.pos = 0

    def tokenize(self, text):
        token_spec = [
            ("AND", r"\bAND\b"),
            ("OR", r"\bOR\b"),
            ("NOT", r"\bNOT\b"),
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("TERM", r"\w+"),
            ("SKIP", r"\s+"),
        ]
        tok_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in token_spec)
        tokens = []
        for mo in re.finditer(tok_regex, text):
            kind = mo.lastgroup
            value = mo.group()
            if kind == "SKIP":
                continue
            tokens.append((kind, value))
        return tokens

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (None, None)

    def consume(self, expected_kind=None):
        token = self.current_token()
        if token[0] is None:
            return token
        if expected_kind and token[0] != expected_kind:
            raise SyntaxError(f"Expected token {expected_kind} but got {token[0]}")
        self.pos += 1
        return token

    def parse(self):
        result = self.parse_expr()
        if self.current_token()[0] is not None:
            raise SyntaxError("Unexpected token at the end")
        return result

    def parse_expr(self):
        node = self.parse_term()
        while True:
            token = self.current_token()
            if token[0] == "OR":
                self.consume("OR")
                right = self.parse_term()
                node = OrNode(node, right)
            else:
                break
        return node

    def parse_term(self):
        node = self.parse_factor()
        while True:
            token = self.current_token()
            if token[0] == "AND":
                self.consume("AND")
                right = self.parse_factor()
                node = AndNode(node, right)
            else:
                break
        return node

    def parse_factor(self):
        token = self.current_token()
        if token[0] == "NOT":
            self.consume("NOT")
            child = self.parse_base()
            return NotNode(child)
        else:
            return self.parse_base()

    def parse_base(self):
        token = self.current_token()
        if token[0] == "LPAREN":
            self.consume("LPAREN")
            node = self.parse_expr()
            self.consume("RPAREN")
            return node
        elif token[0] == "TERM":
            term = self.consume("TERM")[1]
            return TermNode(term)
        else:
            raise SyntaxError("Unexpected token: " + str(token))

class BooleanSearchEngine:
    def __init__(self, inverted_index: PositionalIndex):
        self.inverted_index = inverted_index
        self.documents_dict = inverted_index.get_documents_dict()
        self.all_docs_ids = set(self.documents_dict.keys())

    def search(self, query) -> set[int]:
        """
        Parses and evaluates a Boolean query.
        Returns a list of document (texts).
        """
        parser = BooleanParser(query)
        ast = parser.parse()
        ids = ast.evaluate(self.inverted_index, self.all_docs_ids)
        return [self.documents_dict[doc_id].text for doc_id in ids]

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

if __name__ == '__main__':
    print('\nRunning Boolean search task on cz data from courseware (documents.json)')

    pipeline = pre.PreprocessingPipeline([
        pre.LowercasePreprocessor(),
    ])

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
        print(f'Loading inverted index from file {index_file}')
        index: PositionalIndex = load_index(index_file)
        print(f'Loaded inverted index with {index.get_documents_count()} documents')
    else:
        documents: list[Document] = []
        with open(input_file, 'r', encoding="utf-8") as f:
            tqdm_json = tqdm(json.load(f), desc="Tokenizing documents")
            for doc in tqdm_json:
                documents.append(Document(parse_courseware_cz_demo_document_text(doc)).tokenize())

        tqdm_documents = tqdm(documents)
        for i, doc in enumerate(tqdm_documents, 1):
            tqdm_documents.set_description(f"Processing document {i}/{len(documents)}")
            doc.preprocess(pipeline)

        print('Creating inverted index for preprocessed documents')
        index = PositionalIndex(documents, show_progress=True)

        print('Saving inverted index to file (pickle)')
        print('Note: should be around 1,1GB for the courseware data')
        save_index(index, index_file)

    search_engine = BooleanSearchEngine(inverted_index=index)

    with open(query_file, 'r', encoding="utf-8") as f:
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