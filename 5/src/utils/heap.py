import heapq
from model.document import Document


class HeapEntry:
    def __init__(self, score: float, document: Document):
        self.score = score
        self.document = document

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __gt__(self, other):
        return self.score > other.score

    def __repr__(self):
        return f"({self.score}) {self.document}"


class MinHeap:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.heap: list[HeapEntry] = []

    def push(self, val: HeapEntry):
        heapq.heappush(self.heap, val)

        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)

    def sort(self):
        self.heap.sort(reverse=True)
        return self

    def get_documents(self):
        return [entry.document for entry in self.heap]
