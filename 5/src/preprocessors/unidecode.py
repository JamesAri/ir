from unidecode import unidecode
from model.token import Token
from interface.preprocessor import Preprocessor


class UnidecodePreprocessor(Preprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = unidecode(token.processed_form)
        return token
