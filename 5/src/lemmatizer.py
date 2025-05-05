import stanza
from stanza.pipeline.core import DownloadMethod

from model.document import Document

# stanza.download("cs")
# nlp = stanza.Pipeline("cs", processors="tokenize,lemma")
nlp = stanza.Pipeline(
    "cs",
    processors="tokenize,lemma",
    use_gpu=True,
    download_method=DownloadMethod.REUSE_RESOURCES,
)


def reconstruct_sentence(out_doc):
    lemmatized_sentence = ""
    for sentence in out_doc.sentences:
        for word in sentence.words:
            lemmatized_sentence += word.lemma + " "
    return lemmatized_sentence.strip()


def bulk_lemmatize(documents: list[Document]):
    out_docs = nlp.bulk_process([doc.content for doc in documents])
    zipped_docs = zip(documents, out_docs)
    for doc, out_doc in zipped_docs:
        doc.content = reconstruct_sentence(out_doc)


def lemmatize(document: Document):
    out_doc = nlp.process(document.content)
    document.content = reconstruct_sentence(out_doc)
