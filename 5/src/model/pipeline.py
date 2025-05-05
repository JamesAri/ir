from model.token import Token
from interface.preprocessor import Preprocessor


class PreprocessingPipeline:
    def __init__(self, preprocessors: list[Preprocessor]):
        self.preprocessors = preprocessors

    def preprocess(self, tokens: list[Token], document: str) -> list[Token]:
        for preprocessor in self.preprocessors:
            tokens = preprocessor.preprocess_all(tokens, document)
        return tokens
