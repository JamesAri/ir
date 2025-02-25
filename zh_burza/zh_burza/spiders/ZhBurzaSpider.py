from scrapy import signals
from scrapy.spiders import Spider
from scrapy.http import Request
import sys
from scrapy_playwright.page import PageMethod


from io import StringIO
from html.parser import HTMLParser
from logging.handlers import RotatingFileHandler
from logging import StreamHandler
import logging


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


PAGINATION_TEMPLATE = 'https://www.zatrolene-hry.cz/bazar/?pg={page}'


class ZhBurzaSpider(Spider):
    name = "zh_burza_spider"
    allowed_domains = ["zatrolene-hry.cz"]

    # 500KB
    maxBytes = 500 * 1024
    logging.getLogger().addHandler(RotatingFileHandler('logs/spider.log', maxBytes=maxBytes, backupCount=3))
    logging.getLogger().addHandler(StreamHandler(sys.stdout))

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(ZhBurzaSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def spider_closed(self, spider):
        spider.logger.info("Spider closed: %s", spider.name)
        spider.logger.info("TODO: custom persistence handling")

    def start_requests(self):
        yield Request(
            url='https://www.zatrolene-hry.cz/bazar/',
            callback=self.parse,
        )

    def parse(self, response):
        pagination_visible_numbers = response.css("ul.pagination a.page-link::text").getall()
        last_page_number = int(pagination_visible_numbers[-1])

        for page in range(1, last_page_number + 1):
            yield Request(
                url=PAGINATION_TEMPLATE.format(page=page),
                callback=self.parse_adverts,
            )

    def parse_adverts(self, response):
        for advert in response.css("div[id^='advert_']"):
            link = advert.css("h3 a::attr(href)").get()
            if link:
                full_link = response.urljoin(link)
                yield Request(
                    url=full_link,
                    callback=self.parse_advert_details,
                )
            else:
                self.logger.warning("Link not found")
                continue

    def parse_advert_details(self, response):

        def fmtget(selector):
            return strip_tags(response.css(selector).get(default='')).strip()

        data = {
            'Prodavane_predmety': fmtget("tr:contains('Prodávané předměty') a::text"),
            'Prodavajici': fmtget("tr:contains('Prodávající') a::text"),
            'Stav_hry': fmtget("tr:contains('Stav hry') td:nth-child(2)::text"),
            'Cena': fmtget("tr:contains('Cena') td:nth-child(2)::text"),
            'SafeTrade': fmtget("tr:contains('SafeTrade') td:nth-child(2)"),
            'Prihozu': fmtget("tr:contains('Příhozů') td:nth-child(2)::text"),
            'Nejvyse_prihazujici': fmtget("tr:contains('Nejvýše přihazující') a::text"),
            'Konci_za': fmtget("tr:contains('Končí za') td:nth-child(2)::text"),
            'Moznosti_dopravy': fmtget("tr:contains('Možnosti dopravy') td:nth-child(2)"),
            'Popisek': fmtget("h2 ~ p::text"),
            'Odkaz': response.url,
        }
        yield data
