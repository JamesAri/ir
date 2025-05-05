import os
import preprocessors as pre
from model.pipeline import PreprocessingPipeline

DEFAULT_DATASET = "zh"
DEFAULT_EGINGE = "tf-idf"

TOP_K = 10

DEFAULT_TF_IDF_METHOD = "ltc.ltc"

stopwords_file_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "stopwords",
    "stopwords-cs.txt",
)

SAVE_TO_DISK = True

PIPELINE = PreprocessingPipeline(
    [
        pre.StopWordsPreprocessor(stopwords_file_path),
        pre.LowercasePreprocessor(),
        # pre.HtmlStripPreprocessor(), # done in Document now
        pre.WhitespaceStripPreprocessor(),
        pre.UnidecodePreprocessor(),
    ]
)
