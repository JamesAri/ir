from bs4 import BeautifulSoup
from model.token import Token
from interface.preprocessor import Preprocessor


class HtmlStripPreprocessor(Preprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = BeautifulSoup(
            token.processed_form, "html.parser"
        ).get_text()
        return token
