from model.document import Document
from interface.parser import Parser


class CWParser(Parser):
    @staticmethod
    def parse(document: str) -> Document:
        title = document["title"] or "<missing title>"
        text = document["text"] or "<missing text>"
        return Document(title=title, text=text)
