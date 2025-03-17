from abc import ABC, abstractmethod

from tokenizer import Token, TokenType
from bs4 import BeautifulSoup
from unidecode import unidecode


class TokenPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, token: Token, document: str) -> Token:
        raise NotImplementedError()

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        processed_tokens = []
        for token in tokens:
            processed_token = self.preprocess(token, document)
            if processed_token.processed_form and not processed_token.processed_form.isspace():
                processed_tokens.append(processed_token)
        return processed_tokens


class PreprocessingPipeline:
    def __init__(self, preprocessors: list[TokenPreprocessor]):
        self.preprocessors = preprocessors

    def preprocess(self, tokens: list[Token], document: str) -> list[Token]:
        for preprocessor in self.preprocessors:
            tokens = preprocessor.preprocess_all(tokens, document)
        return tokens

##############################################################################


class LowercasePreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = token.processed_form.lower()
        return token


class HtmlStripPreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = BeautifulSoup(token.processed_form, "html.parser").get_text()
        return token


class StopWordsPreprocessor(TokenPreprocessor):
    def __init__(self, stopwords: set[str]):
        self.stopwords = stopwords

    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form in self.stopwords:
            token.processed_form = ""
        return token


class LemmatizationPreprocessor(TokenPreprocessor):
    """ Note: optimize... => just for demo purposes atm """
    
    def __init__(self):
        # importing stanza can return 500 server unavailable error
        # so we can just comment out this preprocessor if it happens
        import stanza
        stanza.download('cs')
        self.nlp = stanza.Pipeline('cs', processors='tokenize,lemma')

    def preprocess(self, token: Token, document: str) -> Token:
        if token.token_type is TokenType.WORD:
            token.processed_form = self.nlp(token.processed_form).sentences[0].words[0].lemma
        return token


class UnidecodePreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = unidecode(token.processed_form)
        return token


class WhitespaceStripPreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = token.processed_form.replace(" ", "").strip()
        return token

