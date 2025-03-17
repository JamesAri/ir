import re


# AST Node definitions
class Node:
    def evaluate(self, index):
        raise NotImplementedError()


class TermNode(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Term({self.value})"

    def evaluate(self, index):
        # TODO retrieve the value from the index
        return {}


class NotNode(Node):
    def __init__(self, child):
        self.child = child

    def __repr__(self):
        return f"Not({self.child})"

    def evaluate(self, index):
        # TODO apply complement operation on the node
        return {}


class AndNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"And({self.left}, {self.right})"

    def evaluate(self, index):
        # TODO conjunct the left and right nodes
        return {}


class OrNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Or({self.left}, {self.right})"

    def evaluate(self, index):
        # TODO disjunct the left and right nodes
        return {}


class BooleanParser:
    """
    Lexical and syntax analyzer for boolean queries with the following grammar:
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
            # Parse only one NOT by parsing a base after consuming NOT.
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


# Example usage:
if __name__ == "__main__":
    query = "apple AND (banana OR NOT cherry)"
    # query = "(Colorada AND energetiku) OR (NOT meta AND NOT nesetkali AND NOT průzkumu AND znalo AND panely AND NOT jiní) OR (nakupovat AND NOT odpustili AND NOT zahynuli AND navazuje AND NOT odložit AND NOT Prince) OR (NOT odebrání AND hrátky AND NOT spolupracovníka AND NOT visela AND NOT Rychetského) OR (NOT 1981 AND sprejem AND slzy AND NOT napris) OR (NOT Pearce AND vyrážejí AND výzvědnou AND NOT Tomuto AND NOT pochvaluje) OR (NOT otvírání AND musicae AND NOT Asie AND zúčastnění AND NOT vášniví) OR (NOT blýskla AND NOT pobrezi AND NOT nabitými AND NOT prostějovského AND cestování AND ztrácejí) OR (bridge AND Crista AND NOT SÍRY) OR (nevah AND exministra)"
    parser = BooleanParser(query)
    ast = parser.parse()
    print(ast)
