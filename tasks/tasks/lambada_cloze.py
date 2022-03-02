import json
from tasks.base import Task, rf
from tasks.metrics import mean, perplexity
from tasks.utils import sh
from tasks.tasks.lambada import LAMBADA
from best_download import download_file


class LAMBADA_cloze(LAMBADA):
    VERSION = 0
    def doc_to_text(self, doc):
        return doc['text'].rsplit(' ', 1)[0] + " ____. ->"

    def doc_to_target(self, doc):
        return " " + doc['text'].rsplit(' ', 1)[1]
