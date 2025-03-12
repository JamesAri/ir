# spiders/retry_test.py
import scrapy


class RetryTestSpider(scrapy.Spider):
    name = "retry_test"

    def start_requests(self):
        url = 'http://localhost:8000/'  # see server.py from root dir
        yield scrapy.Request(url=url, callback=self.parse)
