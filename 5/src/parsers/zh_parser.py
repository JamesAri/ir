from model.document import Document
from interface.parser import Parser


class ZHParser(Parser):
    @staticmethod
    def parse(document: str) -> Document:
        title = document["Prodavane_predmety"]
        text = document["Popisek"]
        return Document(title=title, text=text)
