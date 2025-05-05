import re
from flask import Flask, render_template, request, redirect, url_for
from markupsafe import Markup
from model.engine_collection import EngineCollection
from model.document import Document
from web.request_manager import RequestManager
from web.session_manager import SessionManager
import config


class HTMLDocument:
    def __init__(self, doc_id: int, title: str, text: str):
        self.doc_id = doc_id
        self.title = title
        self.text = text


class Highlighter:

    def __init__(self, query: str):
        self.terms = re.findall(r"\w+", query)
        print(f"highlight_results - Query terms: {self.terms}")

        escaped_terms = [re.escape(term) for term in self.terms]
        self.pattern = r"\b(" + "|".join(escaped_terms) + r")\b"

    def highlight(self, text: str) -> Markup:
        if not self.terms:
            return Markup(text)

        highlighted_text = re.sub(
            self.pattern, r"<mark>\1</mark>", text, flags=re.IGNORECASE
        )
        return Markup(highlighted_text)

    def highlight_results(self, search_results: list[Document]) -> list[HTMLDocument]:

        highlighted_results = []
        for doc in search_results:
            highlighted_title = self.highlight(doc.title)
            highlighted_text = self.highlight(doc.text)

            highlighted_results.append(
                HTMLDocument(
                    doc_id=doc.doc_id,
                    title=highlighted_title,
                    text=highlighted_text,
                )
            )
        return highlighted_results


class WebServer:
    zh_ec: EngineCollection
    cw_ec: EngineCollection

    def __init__(self, zh_ec: EngineCollection, cw_ec: EngineCollection):
        self.app = Flask(__name__)
        self.app.secret_key = "localhost_only_dummy_key"
        self.zh_ec = zh_ec
        self.cw_ec = cw_ec

    def run(self):
        self.setup_routes()
        self.app.run(debug=False)

    def get_engine_collection(self) -> EngineCollection:
        match SessionManager.get_dataset_name():
            case "zh":
                return self.zh_ec
            case "cw":
                return self.cw_ec
            case _:
                raise ValueError(
                    f"Unknown dataset name: {SessionManager.get_dataset_name()}"
                )

    def search(self, query: str) -> list[HTMLDocument]:
        print(f"Searching with {SessionManager.get_engine_name()} engine")

        ec = self.get_engine_collection()

        highlighter = Highlighter(query)

        match SessionManager.get_engine_name():
            case "lsa_se":
                results = ec.lsa_se.search(query=query, k=config.TOP_K)
                return highlighter.highlight_results(results)
            case "boolean_se":
                results = ec.boolean_se.search(query=query, k=config.TOP_K)
                # TODO
                return highlighter.highlight_results(results)
            case "sent_trans_se":
                results = ec.sent_trans_se.search(query=query, k=config.TOP_K)
                return highlighter.highlight_results(results)
            case "tfidf_se":
                results = ec.tfidf_se.search(query=query, k=config.TOP_K)
                return highlighter.highlight_results(results)
            case _:
                raise ValueError(
                    f"Unknown engine name: {SessionManager.get_engine_name()}"
                )

    def setup_routes(self):
        @self.app.route("/", methods=["GET", "POST"])
        def index():
            query = RequestManager.POST_query()

            if not query:
                return render_template("index.html", query=query)

            results = self.search(query)

            return render_template("index.html", results=results, query=query)

        @self.app.route("/settings", methods=["GET", "POST"])
        def settings():
            if request.method == "POST":
                SessionManager.set_dataset_name(RequestManager.POST_dataset_name())
                SessionManager.set_engine_name(RequestManager.POST_engine_name())
                return redirect(url_for("index"))

            return render_template(
                "settings.html",
                current_dataset=SessionManager.get_dataset_name(),
                current_engine=SessionManager.get_engine_name(),
            )

        @self.app.route("/document/<int:doc_id>")
        def document(doc_id):
            index = self.get_engine_collection().dataset.index
            doc = index.documents_dict[doc_id]

            if not doc:
                return "Document not found", 404

            query = RequestManager.GET_query()

            highlighter = Highlighter(query)

            highlighted_text = highlighter.highlight(doc.text)
            highlighted_title = highlighter.highlight(doc.title)

            return render_template(
                "document.html",
                doc=doc,
                title=highlighted_title,
                text=highlighted_text,
            )

        @self.app.route("/insert", methods=["GET", "POST"])
        def insert():
            if request.method == "POST":
                action = RequestManager.POST_action()
                if action == "insert":
                    title = RequestManager.POST_insert_title()
                    text = RequestManager.POST_insert_text()
                    print(f"Insert title: {title}")
                    print(f"Insert text: {text}")
                    if title and text:
                        doc = (
                            Document(text=title + text, title=title)
                            .tokenize()
                            .preprocess(config.PIPELINE)
                        )
                        engine_collection = self.get_engine_collection()
                        engine_collection.dataset.index.add_document(doc)
                        engine_collection.refresh_engines()

                elif action == "save":
                    print("Save index")
                    self.get_engine_collection().dataset.save_index()
            return render_template("insert.html")
