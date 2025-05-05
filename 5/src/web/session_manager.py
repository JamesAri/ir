from flask import session

import config


class SessionManager:

    @staticmethod
    def get_dataset_name():
        return session.get("dataset", config.DEFAULT_DATASET)

    @staticmethod
    def set_dataset_name(dataset_name):
        session["dataset"] = dataset_name

    @staticmethod
    def get_engine_name():
        return session.get("engine", config.DEFAULT_EGINGE)

    @staticmethod
    def set_engine_name(engine_name):
        session["engine"] = engine_name
