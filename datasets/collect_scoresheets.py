from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from time import sleep
from progressbar import ProgressBar
import lib.config as conf

path_scoresheet_urls = None


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 and
            content_type is not None and
            content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def get_scoresheet(raw_html):
    return raw_html.split(
        """<img onclick="EnlargeThisImage(this);" src=""")[-1].split(
        " style")[0][1:-1]


def main():
    global path_scoresheet_urls
    path_scoresheet_urls = conf.gfc("path_scoresheet_urls")
    ids = range(1, 2800)
    urls_sheets = {}
    with ProgressBar(max_value=len(ids)) as bar:
        for i, idx in enumerate(ids):
            sleep(1)
            url = "http://chessstream.com/ImageToPGNEditor.aspx?ScoresheetImageID={}".format(idx)
            raw_html = simple_get(url)
            if raw_html:
                url_sheet = get_scoresheet(raw_html)
                urls_sheets[idx] = url_sheet
            bar.update(i)
    with open(path_scoresheet_urls, 'w') as f:
        f.write('id,url\n')
        for k, v in urls_sheets.iteritems():
            if len(v) < 200:
                f.write('{},{}\n'.format(k, v))


if "__main__" == __name__:
    conf.loadconfig('./config.global.yaml')
    main()
