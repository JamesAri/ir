## Spusteni

- prvni se musi pridat do slozky `data` soubor `documents.json` a `boolean_queries_simple.txt` z courseware

```bash
python3 -m venv venv && \
source venv/bin/activate && \
pip3 install -r requirements.txt && \
python3 src/parser.py
```

ukazka vystupu: 
- `example_output1.txt`
- `example_output2.txt` (uz s ulozenym invertovanym indexem)

## Ukoly

1. /ukoly/ukol1.jpg
2. /ukoly/ukol2.jpg
3. /src/parser.py
- `[IMPLEMENTED]` Implement an efficient Boolean search engine and test it on the given queries (evaluation data in the term paper tab).
- `[IMPLEMENTED]` Implement an inverted index.
- `[USED THE PROVIDED ONE]` You can use the provided implementation of the recursive descent parser to process the Boolean expression (you don't have to, you can program your own parser).
- `[IMPLEMENTED - LOWERCASE PREPROCESSING ONYL]` Queries should be satisfied by a non-zero number of documents (may not always be true, depends on preprocessing).
- `[IMPLEMENTED]` Try to minimize the computation time of your search engine (by optimizations).
