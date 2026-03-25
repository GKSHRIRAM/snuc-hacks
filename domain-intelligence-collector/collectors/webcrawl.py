import scrapy
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor, defer
import os
import json

class DomainSpider(scrapy.Spider):
    name = 'domain_spider'
    
    def start_requests(self):
        yield scrapy.Request(
            f"http://{self.domain}", 
            meta=dict(
                playwright=True,
                playwright_include_page=True,
                errback=self.errback,
            )
        )

    async def parse(self, response):
        page = response.meta["playwright_page"]
        title = await page.title()
        content = await page.content()
        await page.close()
        
        yield {
            "url": response.url,
            "title": title,
            "content_length": len(content)
        }

    async def errback(self, failure):
        page = failure.request.meta["playwright_page"]
        await page.close()

def run_spider(domain, output_file):
    # This is a synchronous function that runs the reactor.
    # It must be run in a separate process in Flask usually,
    # because twisted reactor cannot be restarted.
    from scrapy.crawler import CrawlerProcess
    
    settings = {
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'TWISTED_REACTOR': "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        'PLAYWRIGHT_LAUNCH_OPTIONS': {"headless": True},
        'FEEDS': {
            output_file: {
                'format': 'json',
                'overwrite': True
            }
        },
        'LOG_LEVEL': 'ERROR'
    }
    
    process = CrawlerProcess(settings)
    process.crawl(DomainSpider, domain=domain)
    process.start()

async def collect_webcrawl(domain):
    # To run scrapy in an async api without reactor crash,
    # we run it via subprocess
    import sys
    import subprocess
    
    output_path = os.path.join("outputs", "webcrawl.json")
    
    # Create a temporary runner script
    runner_script = "run_scrapy_temp.py"
    with open(runner_script, "w") as f:
        f.write(f'''
import sys
from collectors.webcrawl import run_spider
run_spider("{domain}", "{output_path}")
        ''')
    
    process = await asyncio.create_subprocess_exec(
        sys.executable, runner_script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    await process.communicate()
    
    if os.path.exists(runner_script):
        os.remove(runner_script)
        
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

import asyncio
