from flask import request

import config


class RequestManager:
    @staticmethod
    def GET_query():
        return request.args.get("query", "")

    @staticmethod
    def POST_query():
        return request.form.get("query", "")

    @staticmethod
    def POST_action():
        return request.form.get("action", "")

    @staticmethod
    def GET_document_id():
        return request.args.get("doc_id", None)

    @staticmethod
    def POST_dataset_name():
        return request.form.get("dataset", config.DEFAULT_DATASET)

    @staticmethod
    def POST_engine_name():
        return request.form.get("engine", config.DEFAULT_EGINGE)

    @staticmethod
    def POST_insert_title():
        return request.form.get("title", "").strip()

    @staticmethod
    def POST_insert_text():
        return request.form.get("text", "").strip()
