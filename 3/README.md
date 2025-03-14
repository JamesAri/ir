## Spusteni

```bash
python3 -m venv venv && \
source venv/bin/activate && \
pip3 install -r requirements.txt && \
python3 src/main.py
```

## Ukoly

1. /ukoly/ukol1.jpg
2. /ukoly/ukol2.txt
3. /src
	- invertovany seznam
	- podpora xxx.yyy metod
	- ... viz potencionalni body
4. /src/main.py
	- po spustnei se vypise "Running 1." popr. "Running 2." s vysledky
5. /src/main.py
	- po spusteni se vypise "Running 5."

## Potencionalni body
- podpora xxx.yyy zapisu (viz prednaska se slidem `Components of tf-idf weighting`)
	- aktualne pouzivne `ltu.ltc` a `ltc.ltc`
- nad daty z 1. cviceni delam pivoted normalization
- query pro data z 1. cviceni hardcoded v `main.py`
- funguje s programem z 2. cviceni
- pouziti positional indexu (invertovaný seznam s postingy)
- pocitani pouze nad relevantnimi dokumenty
- ruzne optimalizace
- search vraci top `k` (vyuziti MinHeapu)
- ...

## Data

Sample je zatim maly (vzdy ho prikladam do zipu spolu se zdrojaky), ale k semestralce vytvorim vetsi datovou sadu spolu s ulozistem, ktere nasdilim.