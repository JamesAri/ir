from itemadapter import is_item, ItemAdapter
from twisted.internet import reactor
from scrapy import signals
import random
from fake_useragent import UserAgent
from twisted.internet.task import deferLater
from scrapy.downloadermiddlewares.retry import RetryMiddleware


def sleep(seconds):
    return deferLater(reactor, seconds, lambda: None)


class ExponentialBackoffRetryMiddleware(RetryMiddleware):
    def __init__(self, settings):
        super().__init__(settings)
        self.max_delay = settings.getfloat('RETRY_BACKOFF_MAX', 60)

    def _retry(self, request, reason, spider):
        retries = request.meta.get('retry_times', 0) + 1
        delay = min((2 ** retries) + random.uniform(0, 1), self.max_delay)  # TODO: use grpc approach
        spider.logger.info(f"Retrying {request.url} in {delay:.2f} seconds (Retry #{retries})")
        yield sleep(delay)
        return super()._retry(request, reason, spider)


class RandomUserAgentMiddleware:
    def __init__(self):
        self.ua = UserAgent()

    def process_request(self, request, spider):
        # Randomize User-Agent
        user_agent = self.ua.random
        request.headers['User-Agent'] = user_agent

        # Randomize Accept-Language
        languages = [
            'en-US,en;q=0.9',       # English (US)
            'en-GB,en;q=0.8',       # English (UK)
            'fr-FR,fr;q=0.7',       # French
            'cs-CZ,cs;q=0.9',       # Czech
            'sk-SK,sk;q=0.9',       # Slovak
            'de-DE,de;q=0.9'        # German
        ]
        request.headers['Accept-Language'] = random.choice(languages)

        # Additional headers
        # request.headers['Accept-Encoding'] = 'gzip, deflate, br' # TODO
        request.headers['Connection'] = 'keep-alive'


# ##### TEMPLATES #####

class ZhBurzaSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class ZhBurzaDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)
