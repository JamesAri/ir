## Spusteni

- prvni se musi pridat do slozky `data` soubor `documents.json` z courseware, defaultne je tam muj dataset

```bash
python3 -m venv venv && \
source venv/bin/activate && \
pip3 install -r requirements.txt && \
python3 src/semantic.py
```

ukazka vystupu pro 3 vyhledavace - If-IDf, s pouzitim LSA a s pouzitim sentence transformers: 
- `results/custom-data.txt` - na vlastnich datech
- `results/courseware-data.txt` - na courseware datech

**pozn.: ukladani pickled indexu, prvni beh pomale, ale potom uz je to radove rychlejsi... zabere cca 1.1GB**
## Ukoly

- implementovano LSA pro redukci dimenzionality, pouze maticove operace, vraci pomerne stejne vysledky jako Tf-IDf vyhledavac z predesleho ukolu
	- omezeny scope za cenu rychlosti, ale zkousel jsem pro celou kolekci dokumentu a dava to pekne vysledky
	- bohuzel oproti Tf-IDf relativne pomale
- vyuziti sentence transformers a modelu `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
	- bud velmi velmi velmi dobre vysledky a nebo naprosto nerelevantni vysledek, ale jinak opravdu super, jednoduche
	- casove nejnarocnejsi metoda
- kombinace vysledku (top 5 z Tf-IDf, top 5 z LSA, top 5 z Transformers)
