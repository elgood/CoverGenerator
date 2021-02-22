import argparse
from bing_image_downloader import downloader
from args import get_default_ArgumentParser, process_common_arguments


def main():

  message  =("Downloads a bunch of images from bing " +
    "based on a search query.")  
  parser = get_default_ArgumentParser(message)
  parser.add_argument("--query", type=str, default="dragon",
    help="The query that is used to get the images.")
  parser.add_argument("destination", type=str,
    help="The directory where the files should be placed.")
  parser.add_argument("--limit", type=int, default=1000,
    help="The number of images to get.")

  FLAGS = parser.parse_args()
  process_common_arguments(FLAGS)

  downloader.download(FLAGS.query, 
                      output_dir=FLAGS.destination, 
                      limit=FLAGS.limit)

if __name__ == '__main__':
  main()
