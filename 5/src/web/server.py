import os

from model.dataset import Dataset
from model.engine_collection import EngineCollection
from parsers import CWParser, ZHParser

from .handlers import WebServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "out"))


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def run():
    ensure_out_dir()

    ZH_DATASET = Dataset(
        json_file=os.path.join(BASE_DIR, "..", "..", "data", "zh.json"),
        pickle_file=os.path.join(OUT_DIR, "zh_index.pkl"),
        parser=ZHParser,
        tag="Zatrolene hry",
    )

    CW_DATASET = Dataset(
        json_file=os.path.join(BASE_DIR, "..", "..", "data", "cw.json"),
        pickle_file=os.path.join(OUT_DIR, "cw_index.pkl"),
        parser=CWParser,
        tag="Courseware data",
    )

    server = WebServer(
        zh_ec=EngineCollection(dataset=ZH_DATASET),
        cw_ec=EngineCollection(dataset=CW_DATASET),
    )
    server.run()
