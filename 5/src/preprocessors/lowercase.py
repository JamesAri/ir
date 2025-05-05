from model.token import Token
from interface.preprocessor import Preprocessor


class LowercasePreprocessor(Preprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = token.processed_form.lower()
        return token
