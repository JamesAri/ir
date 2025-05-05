from model.token import Token
from interface.preprocessor import Preprocessor


class WhitespaceStripPreprocessor(Preprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = token.processed_form.replace(" ", "").strip()
        return token
