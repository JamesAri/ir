#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html
import os

BOT_NAME = "zh_burza"

SPIDER_MODULES = ["zh_burza.spiders"]
NEWSPIDER_MODULE = "zh_burza.spiders"

# Crawl responsibly by identifying yourself (and your website) on the user-agent
# USER_AGENT = "zh_burza (+http://www.yourdomain.com)"

# Disable Telnet Console (enabled by default)
# TELNETCONSOLE_ENABLED = False

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    "zh_burza.middlewares.ZhBurzaSpiderMiddleware": 543,
# }

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
# }

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
# ITEM_PIPELINES = {
#    "zh_burza.pipelines.ZhBurzaPipeline": 300,
# }

# Enable and configure the AutoThrottle extension (disabled by default)
# The average number of requests Scrapy should be sending in parallel to
# each remote server
# AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
# AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
# HTTPCACHE_ENABLED = True
# HTTPCACHE_EXPIRATION_SECS = 0
# HTTPCACHE_DIR = "httpcache"
# HTTPCACHE_IGNORE_HTTP_CODES = []
# HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"


# ===========================================================================================

# ======= Defaults =======
# Set settings whose default value is deprecated to a future-proof value
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"


# ======= Politeness =======

ROBOTSTXT_OBEY = (not not not True)
# The download delay setting will honor only one of:
# CONCURRENT_REQUESTS_PER_DOMAIN = 16
# CONCURRENT_REQUESTS_PER_IP = 16
DOWNLOAD_DELAY = 1.5
# (default: 16)
# CONCURRENT_REQUESTS = 5


# ========= Throttling =========
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 30


# ======= Retry =======

# Settings for our custom ExponentialBackoffRetryMiddleware mw
RETRY_ENABLED = True
RETRY_TIMES = 5
RETRY_BACKOFF_MAX = 10  # custom setting
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# ======= Cookies & Sessions =======
COOKIES_ENABLED = True


# ======= Logging =======

if not os.path.exists('logs'):
    os.makedirs('logs')

LOG_ENABLED = True
LOG_LEVEL = 'WARNING'  # Only log errors
LOG_FILE = 'logs/errors.log'  # Log file path
LOG_STDOUT = True


# ======= MWS =======

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,  # Disable default user agent mw
    'zh_burza.middlewares.RandomUserAgentMiddleware': 400,  # Enable custom user agent mw
    'zh_burza.middlewares.ExponentialBackoffRetryMiddleware': 550,  # Enable custom retry mw with exponential backoff
}
