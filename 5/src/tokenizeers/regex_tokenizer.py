import re
from model.token import Token, TokenType
from interface.tokenizer import Tokenizer

regex_map = {
    TokenType.URL: r"(http\S+|www\S+)",
    # TokenType.TAG: r"(<.*?>)", # handeled in Documents
    TokenType.EDITION: r"\d+(?:\.)?(?:th|nd|rd|st)?\s*(?:edition|edice|edici|vydani|vydání|vydanie|vyd\.|díl|dil|sérii|serii)\w*",
    TokenType.EXTENSION: r"\d+(?:\.)?(?:th|nd|rd|st)?\s*(?:rozšíření|rozsireni|rozš|rozs)\w*",
    TokenType.NUMBER: r"(\d+[.,]?\d*)",
    TokenType.WORD: r"(\w{2,})",
    TokenType.PUNCT: r"([^\w\s]+)",
}

tokenization_order = [
    TokenType.URL,
    # TokenType.TAG, # handeled in Documents
    TokenType.EDITION,
    TokenType.EXTENSION,
    TokenType.NUMBER,
    TokenType.WORD,
    TokenType.PUNCT,
]

pattern = "|".join(
    [
        f"(?P<{token_type.name}>{regex_map[token_type]})"
        for token_type in tokenization_order
    ]
)


class RegexMatchTokenizer(Tokenizer):

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        for match in re.finditer(
            re.compile(pattern, re.UNICODE | re.IGNORECASE), document
        ):
            for token_type in tokenization_order:
                token = match.group(token_type.name)
                if token:
                    if token_type == TokenType.EDITION:
                        num = re.search(r"\d+", token).group()
                        token = f"{num}[ed]"
                    elif token_type == TokenType.EXTENSION:
                        num = re.search(r"\d+", token).group()
                        token = f"{num}[ex]"
                    elif token_type == TokenType.NUMBER:
                        token = "[num]"
                    tokens.append(
                        Token(
                            token,
                            match.start(),
                            match.end() - match.start(),
                            token_type,
                        )
                    )
                    break
        return tokens
