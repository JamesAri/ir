## How to run

```bash
python3 -m venv venv && \
source venv/bin/activate && \
pip3 install -r requirements.txt && \
python3 src/main.py
```

## Potential points

- tokenization
	- `URL`
	- `TAG`
	- `EDITION` (number + text value)
	- `EXTENSION` (number + text value)
	- `NUMBER` (further preprocessed)
	- `WORD`
	- `PUNCT`
- preprocessing
	- html strip with `bs4`
	- filtering by token type
	- czech stopwords filtering
	- czech lemmatization with `stanza` nlp
	- lowercase preproc
	- unidecode (diacritics removal)
	- whitespace normalization
	- token len filter
- progress bar :-)

