import re

from model.positional_index import PositionalIndex
from model.document import Document
from lemmatizer import lemmatize
import config


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
        return self.left.evaluate(index, all_docs_ids) & self.right.evaluate(
            index, all_docs_ids
        )


class OrNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Or({self.left}, {self.right})"

    def evaluate(self, index: PositionalIndex, all_docs_ids: list[int]) -> set[int]:
        """Apply union: A OR B → A u B"""
        return self.left.evaluate(index, all_docs_ids) | self.right.evaluate(
            index, all_docs_ids
        )


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
        print(f"Tokens: {self.tokens}")
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
            raise SyntaxError(
                f"Unexpected token '{self.current_token()[0]}' at the end"
            )
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
            term_doc = Document(text=term)
            lemmatize(term_doc)
            term_doc = term_doc.tokenize().preprocess(config.PIPELINE)
            term_text = term_doc.tokens[0].processed_form if term_doc.tokens else term
            return TermNode(term_text)
        else:
            raise SyntaxError("Unexpected token: " + str(token))
