from pathlib import Path
import os
import sys
import urllib.request
import urllib
import imghdr
import posixpath
import re
import logging
import hashlib

'''
Python api to download image form Bing.
Author: Guru Prasad (g.gaurav541@gmail.com)
'''

class Bing:
  def __init__(self, query, limit, output_dir, adult, timeout, filters=''):
    self.query = query
    self.output_dir = output_dir
    self.adult = adult
    self.filters = filters
    self.digests = set()

    assert type(limit) == int, "limit must be integer"
    self.limit = limit
    assert type(timeout) == int, "timeout must be integer"
    self.timeout = timeout

    self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}

    # Reading all the images in the output_dir so that we don't try to add them again
    for filename in os.listdir(output_dir):
      with open(os.path.join(output_dir, filename), "rb") as f:
        content = f.read()
        md5 = hashlib.md5()
        md5.update(content)
        digest = md5.hexdigest()
        self.digests.update([digest])
    
    self.download_count = len(self.digests)
    logging.info("Initial count " + str(self.download_count))

  def download_image(self, link):

    # Get the image link
    try:
      path = urllib.parse.urlsplit(link).path
      filename = posixpath.basename(path).split('?')[0]
      file_type = filename.split(".")[-1]
      if file_type.lower() not in ["jpe", "jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
        file_type = "jpg"

      # Download the image
      logging.info("[%] Downloading Image #{} from {}".format(self.download_count, link))


      request = urllib.request.Request(link, None, self.headers)
      image = urllib.request.urlopen(request, timeout=self.timeout).read()
      if not imghdr.what(None, image):
        logging.warn('[Error]Invalid image, not saving {}\n'.format(link))
        raise

      md5 = hashlib.md5()
      md5.update(image)
      digest = md5.hexdigest()
      if digest not in self.digests:
        self.download_count += 1
        file_path = os.path.join(self.output_dir, 
          "Image_{}.{}".format(str(self.download_count), file_type))
        logging.info("Adding " + file_path)

        with open(file_path, 'wb') as f:
          f.write(image)

        self.digests.update([digest])

      else:
        logging.info("Ignoring " + link + " because the hash exists. " +
          "num_unique " + str(self.download_count))

    except Exception as e:
      logging.warn("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))

  def run(self):
    step = 10
    for i in range(1, self.limit, step):
      logging.info('\n\n[!!]Indexing page: {}\n'.format(i))
      # Parse the page source and download pics
      request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(self.query) \
             + '&first=' + str(i) + '&count=' + str(step) \
             + '&adlt=' + self.adult + '&qft=' + self.filters
      logging.info("Request url: " + request_url)
      request = urllib.request.Request(request_url, None, headers=self.headers)
      response = urllib.request.urlopen(request)
      html = response.read().decode('utf8')
      links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)

      logging.info("[%] Indexed {} Images on Page {}.".format(len(links), str(i)))
      logging.info("\n===============================================\n")

      for link in links:
        self.download_image(link)
        logging.info("\n\n[%] Done. Downloaded {} images.".format(self.download_count))
        logging.info("\n===============================================\n")

