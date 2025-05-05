from model.token import Token
from interface.preprocessor import Preprocessor
from utils.stopwords import load_stopwords


class StopWordsPreprocessor(Preprocessor):
    def __init__(self, file_path: str):
        self.stopwords = load_stopwords(file_path=file_path)

    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form in self.stopwords:
            token.processed_form = ""
        return token
