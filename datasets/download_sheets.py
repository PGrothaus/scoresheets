import pandas as pd
from PIL import Image
import urllib
import cStringIO
import os
from progressbar import ProgressBar
import lib.config as conf

path_scoresheet_urls = None
dir_scoresheets = None


def download_scoresheets():
    df = pd.read_csv(path_scoresheet_urls)
    urls = df['url']
    with ProgressBar(max_value=len(urls)) as bar:
        for i, url in enumerate(urls):
            tail = url.split('/')[-1]
            dl_and_save_image(url, dir_scoresheets, tail)
            bar.update(i)


def dl_and_save_image(url, destination_folder, outfile=''):
    # type: (str, str, str) -> None
    if not outfile:
        path = os.path.join(destination_folder, url.split('/')[-1])
    else:
        path = os.path.join(destination_folder, outfile)
    if not os.path.isfile(path):
        img = dl_image(url)
        if img:
            img.save(path)


def dl_image(url):
    # type: (str) -> Optional[Image]
    if url:
        try:
            file = cStringIO.StringIO(urllib.urlopen(url).read())
        except IOError:
            print 'Image could not be downloaded:\n     {}'.format(url)
            return None
        try:
            img = Image.open(file).convert('RGB')
        except IOError:  # Content deleted
            img = None
    else:
        img = None
    return img


def main():
    global path_scoresheet_urls
    global dir_scoresheets
    path_scoresheet_urls = conf.gfc("path_scoresheet_urls")
    dir_scoresheets = conf.gfc("dir_scoresheets")
    download_scoresheets()


if "__main__" == __name__:
    conf.loadconfig('./config.global.yaml')
    main()
