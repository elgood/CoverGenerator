import scrapy
from scrapy.linkextractors import LinkExtractor

"""
  Usage: scrapy crawl images -a search_string=<search string>
  (from imagescraper directory)

  You have to specify the output directory of the images in settings.py
  (one level up).
"""


class ImageItem(scrapy.Item):
  image_urls = scrapy.Field()
  images = scrapy.Field()


class ImageScraper(scrapy.Spider):
  name = "images"

  def __init__(self, search_string="", outdir="",
               *args, **kwargs):
    super(ImageScraper, self).__init__(*args, **kwargs)

    if search_string == "":
      self.log(f'Warning: search string not specified')
    if outdir == "":
      self.log(f'Warning: outdir not specified')
    else:
      custom_settings["IMAGES_STORE"] = outdir
   
    self.outdir = outdir 
    self.log(f'Search string {search_string}')
    url = f"https://google.com/search?q={search_string}"
    self.start_urls = [ url ]

  def start_requests(self):
    for url in self.start_urls:
      yield scrapy.Request(url=url, callback=self.parse)

  def parse(self, response):
    xlink = LinkExtractor()
    for link in xlink.extract_links(response):
       
      for elem in response.xpath("//img"):
        img_url = elem.xpath("@src").extract_first()
        yield ImageItem(image_urls=[img_url])

      self.log(link)
      yield scrapy.Request(url=link.url, callback=self.parse)
    #page = response.url.split("/")[-2]
    #filename = f'{self.outdir}/test-{page}.html'
    #with open(filename, 'wb') as f:
    #  f.write(response.body)
    #self.log(f'Saved file {filename}')

    #next_page_url = response.css("li.next > a::attr(href)").extract_first()
    #self.log(f"next_page_url {next_page_url}")
    #if next_page_url is not None:
    #  yield scrapy.Request(url=response.urljoin(next_page_url), callback=self.parse)
