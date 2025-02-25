### How to run:

```bash
pip install -r requirements.txt
cd zh_burza
scrapy crawl zh_burza_spider -o data/output.json -s JOBDIR=crawls/zh_burza_job
```

### Data sample (from crawled web):

- sample.json in root of the project

### Potential points

- "complex" pagination (no `next` button, it's like 1, 2, 3, ..., 14) [see here](https://www.zatrolene-hry.cz/bazar/)
- custom exponential backoff retry middleware
- randomized user-agents from fake_useragent db (not being banned/detected)
	- TODO: proxy rotation (using Tor/paid service), TLS fingerprinting, randomized delays between requests, handle Cookies & Sessions, randomize URLs crawling, ...
	- TODO: will be used for login 
- logging -> `logs/errors.log`, rotating logs (`logs/spider.log.1`, `logs/spider.log.2`, ...), `stdout`
- jobs (Perzistentní stav aplikace)
- collect links on a page -> go to next page -> collect linkgs on a page... -> parse links (Složitá struktura dat)
- early submit
- easy (scrapy settings only):
	- throttling
	- cookies persistence
	- ...

