from abc import ABC, abstractmethod
from model.token import Token


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, token: Token, document: str) -> Token:
        raise NotImplementedError()

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        processed_tokens = []
        for token in tokens:
            processed_token = self.preprocess(token, document)
            if (
                processed_token.processed_form
                and not processed_token.processed_form.isspace()
            ):
                processed_tokens.append(processed_token)
        return processed_tokens
