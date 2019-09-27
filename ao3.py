# coding: utf-8

import re
import os
import sys
import json
import csv
import random
import argparse
import requests
import collections
from time import sleep

import numpy
import pandas as pd
from bs4 import BeautifulSoup

import search
import vis

try:
    import lextrie
    bing = lextrie.LexTrie.from_plugin('bing')

    try:
        emolex = lextrie.LexTrie.from_plugin('emolex_en')
    except Exception:
        emolex = None

    try:
        liwc = lextrie.LexTrie.from_plugin('liwc')
    except Exception:
        liwc = None
except ImportError:
    bing = None
    emolex = None
    liwc = None


# -----------------------------------------------------------------------------
# HTML TO TXT FUNCTIONS
# ---------------------

def get_fan_work(fan_html_name):
    with open(fan_html_name, encoding='utf8') as fan_in:
        fan_html = BeautifulSoup(fan_in.read(), "lxml")
        fan_txt = fan_html.find(id='workskin')
        if fan_txt is None:
            return ''

    fan_txt = ' '.join(fan_txt.strings)
    fan_txt = re.split(r'Work Text\b([\s:]*)', fan_txt, maxsplit=1)[-1]
    fan_txt = re.split(r'Chapter 1\b([\s:]*)', fan_txt, maxsplit=1)[-1]
    fan_txt = fan_txt.replace('Chapter Text', ' ')
    fan_txt = re.sub(r'\s+', ' ', fan_txt).strip()
    return fan_txt

def convert_dir(args):
    html_dir = args.input
    out_dir = args.output

    try:
        os.makedirs(out_dir)
    except Exception:
        pass

    errors = []
    for infile in os.listdir(html_dir):
        base, ext = os.path.splitext(infile)
        outfile = os.path.join(out_dir, base + '.txt')
        infile = os.path.join(html_dir, infile)

        if not os.path.exists(outfile):
            text = get_fan_work(infile)
            if text:
                with open(outfile, 'w', encoding='utf-8') as out:
                    out.write(text)
            else:
                errors.append(infile)

    error_outfile = 'clean-html-errors.txt'
    with open(error_outfile, 'w', encoding='utf-8') as out:
        out.write('The following files were not converted:\n\n')
        for e in errors:
            out.write(e)
            out.write('\n')

# ------------------
# METADATA FUNCTIONS
# ------------------

def select_text(soup_node, selector):
    sel = soup_node.select(selector)
    return sel[0].get_text().strip() if sel else 'AOOO_UNSPECIFIED'
    # "AOOO_UNSPECIFIED" means value not in An Archive of Our Own metadata field

meta_headers = ['FILENAME', 'TITLE', 'AUTHOR', 'SUMMARY', 'NOTES',
                'PUBLICATION_DATE', 'LANGUAGE', 'TAGS']
def get_fan_meta(fan_html_name):
    with open(fan_html_name, encoding='utf8') as fan_in:
        fan_html = BeautifulSoup(fan_in.read(), 'lxml')

    title = select_text(fan_html, '.title.heading')
    author = select_text(fan_html, '.byline.heading')
    summary = select_text(fan_html, '.summary.module')
    notes = select_text(fan_html, '.notes.module')
    date = select_text(fan_html, 'dd.published')
    language = select_text(fan_html, 'dd.language')
    tags = {k.get_text().strip().strip(':'):
            v.get_text(separator='; ').strip().strip('\n; ')
            for k, v in
            zip(fan_html.select('dt.tags'), fan_html.select('dd.tags'))}
    tags = json.dumps(tags)

    path, filename = os.path.split(fan_html_name)

    vals = [filename, title, author, summary, notes,
            date, language, tags]
    return dict(zip(meta_headers, vals))

def collect_meta(args):
    in_dir = args.input
    out_file = args.output

    errors = []
    rows = []
    for infile in os.listdir(in_dir):
        infile = os.path.join(in_dir, infile)
        rows.append(get_fan_meta(infile))

    error_outfile = out_file + '-errors.txt'
    with open(error_outfile, 'w', encoding='utf-8') as out:
        out.write('Metadata could not be collected from the following files:\n\n')
        for e in errors:
            out.write(e)
            out.write('\n')

    csv_outfile = out_file + '.csv'
    with open(csv_outfile, 'w', encoding='utf-8') as out:
        wr = csv.DictWriter(out, fieldnames=meta_headers)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

#----------------
#SCRAPE FUNCTIONS
#----------------
class Logger:
    def __init__(self, logfile='log.txt'):
        self.logfile = logfile

    def log(self, msg, newline=True):
        with open(self.logfile, 'a') as f:
            f.write(msg)
            if newline:
                f.write('\n')

_logger = Logger()
log = _logger.log

_error_id_log = Logger(logfile='error-ids.txt')
log_error_id = _error_id_log.log

def load_error_ids():
    with open(_error_id_log.logfile, 'w+') as ip:
        ids = set(l.strip() for l in ip.readlines())
        return ids

class InlineDisplay:
    def __init__(self):
        self.currlen = 0

    def display(self, s):
        print(s, end=' ')
        sys.stdout.flush()
        self.currlen += len(s) + 1

    def reset(self):
        print('', end='\r')
        print(' ' * self.currlen, end='\r')
        sys.stdout.flush()
        self.currlen = 0

_id = InlineDisplay()
display = _id.display
reset_display = _id.reset

def request_loop(url, timeout=4.0, sleep_base=1.0):
    # We try 20 times. But we double the delay each time,
    # so that we don't get really annoying. Eventually the
    # delay will be more than an hour long, at which point
    # we'll try a few more times, and then give up.

    orig_url = url
    for i in range(20):
        if sleep_base > 7200:  # Only delay up to an hour.
            sleep_base /= 2
            url = '{}#{}'.format(orig_url, random.randrange(1000))
        display('Sleeping for {} seconds;'.format(sleep_base))
        sleep(sleep_base)
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError:
            code = response.status_code
            if code >= 400 and code < 500:
                display('Unrecoverable error ({})'.format(code))
                return ''
            else:
                sleep_base *= 2
                display('Recoverable error ({});'.format(code))
        except requests.exceptions.ReadTimeout as exc:
            sleep_base *= 2
            display('Read timed out -- trying again;')
        except requests.exceptions.RequestException as exc:
            sleep_base *= 2
            display('Unexpected error ({}), trying again;\n'.format(exc))
    else:
        return None

def scrape(args):
    search_term = args.search
    tag = args.tag
    header = args.url
    out_dir = args.out
    end = args.startpage

    # tag scraping option
    if search_term:
        pp = 1
        safe_search = search_term.replace(' ', '+')
        # an alternative here is to scrape this page and use regex to filter the results:
        # http://archiveofourown.org/media/Movies/fandoms?
        # the canonical filter is used here because the "fandom" filter on the
        # beta tag search is broken as of November 2017
        search_ref = "http://archiveofourown.org/tags/search?utf8=%E2%9C%93&query%5Bname%5D=" + safe_search + "&query%5Btype%5D=&query%5Bcanonical%5D=true&page="
        print('\nTags:')

        tags = ["initialize"]
        while (len(tags)) != 0:
            results_page = requests.get(search_ref + str(pp))
            results_soup = BeautifulSoup(results_page.text, "lxml")
            tags = results_soup(attrs={'href': re.compile('^/tags/[^s]....[^?].*')})

            for x in tags:
                print(x.string)

            pp += 1

    # fan work scraping options
    if header or tag:
        try:
            os.makedirs(out_dir)
        except Exception:
            pass

        os.chdir(out_dir)
        error_works = load_error_ids()

        results = ["initialize"]
        while (len(results)) != 0:
            log('\n\nPAGE ' + str(end))
            print('Page {} '.format(end))

            display('Loading table of contents;')

            if tag:
                mod_header = tag.replace(' ', '%20')
                header = "http://archiveofourown.org/tags/" + mod_header + "/works"

            request_url = header + "?page=" + str(end)
            toc_page = request_loop(request_url)
            if not toc_page:
                err_msg = 'Error loading TOC; aborting.'
                log(err_msg)
                display(err_msg)
                reset_display()
                continue

            toc_page_soup = BeautifulSoup(toc_page, "lxml")
            results = toc_page_soup(attrs={'href': re.compile('^/works/[0-9]+[0-9]$')})

            log('Number of Works on Page {}: {}'.format(end, len(results)))
            log('Page URL: {}'.format(request_url))
            log('Progress: ')

            reset_display()

            for x in results:
                body = str(x).split('"')
                docID = str(body[1]).split('/')[2]
                filename = str(docID) + '.html'

                if os.path.exists(filename):
                    display('Work {} already exists -- skpping;'.format(docID))
                    reset_display()
                    msg = ('skipped existing document {} on '
                           'page {} ({} bytes)')
                    log(msg.format(docID, str(end),
                                   os.path.getsize(filename)))
                elif docID in error_works:
                    display('Work {} is known to cause errors '
                            '-- skipping;'.format(docID))
                    reset_display()
                    msg = ('skipped document {} on page {} '
                           'known to cause errors')
                    log(msg.format(docID, str(end)))

                else:
                    display('Loading work {};'.format(docID))
                    work_request_url = "https://archiveofourown.org/" + body[1] + "?view_adult=true&view_full_work=true"
                    work_page = request_loop(work_request_url)

                    if work_page is None:
                        error_works.add(docID)
                        log_error_id(docID)
                        continue

                    with open(filename, 'w', encoding='utf-8') as html_out:
                        bytes_written = html_out.write(str(work_page))

                    msg = 'reached document {} on page {}, saved {} bytes'
                    log(msg.format(docID, str(end), bytes_written))
                    reset_display()

            reset_display()
            end = end + 1

# -----------------------------------
# data visualization format functions
# -----------------------------------
def project_sentiment_keys_shortform(counts, keys):
        counts = [{k: ct.get(k, 0) for k in keys}
                  for ct in counts]
        for ct in counts:
            if sum(ct.values()) == 0:
                ct['UNDETERMINED'] = 1
            else:
                ct['UNDETERMINED'] = 0
        return counts

def regex(name):
    return (re.sub('(?P<name> (\w))*(\\(.*\\))', '\g<name>', name)).strip()

def format_data(args):
    original_script_markup = args.script
    match_table = args.matches
    output = args.output

    matches = pd.read_csv(match_table)

    name = 'Frequency of Reuse (Exact Matches)'
    positive_match = matches.BEST_COMBINED_DISTANCE <= 0
    matches_thresh = matches.assign(**{name: positive_match})

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    threshname = ['Frequency of Reuse (0-{})'.format(str(t)) for t in thresholds]
    for thresh, name in zip(thresholds, threshname):
        positive_match = matches.BEST_COMBINED_DISTANCE <= thresh
        matches_thresh = matches_thresh.assign(**{name: positive_match})
    thresholds = [0] + thresholds
    threshname = ['Frequency of Reuse (Exact Matches)'] + threshname

    os_markup_raw = search.load_markup_script(original_script_markup)
    os_markup_header = os_markup_raw[0]
    os_markup_raw = os_markup_raw[1:]

    lt = emolex # LexTrie.from_plugin('emolex_en')
    emo_terms = ['ANGER',
                 'ANTICIPATION',
                 'DISGUST',
                 'FEAR',
                 'JOY',
                 'SADNESS',
                 'SURPRISE',
                 'TRUST',
                 'NEGATIVE',
                 'POSITIVE']

    os_markup_header.extend(emo_terms)
    for r in os_markup_raw:
        emos = lt.get_lex_tags(r[0])
        r.extend(int(t in emos) for t in emo_terms)

    os_markup = pd.DataFrame(os_markup_raw, columns=os_markup_header)
    os_markup.index.name = 'ORIGINAL_SCRIPT_WORD_INDEX'

    os_markup.CHARACTER = os_markup.CHARACTER.apply(regex)
    top_eight = collections.Counter(os_markup.CHARACTER).most_common(8)
    top_eight_list = []
    for (name_top,val) in top_eight:
        top_eight_list = top_eight_list + [name_top]
    top_eight_char = ["CHARACTER_" + name for name in top_eight_list]
    used_names = []
    name_char = top_eight[0][0]
    positive_match = 1 * (os_markup.CHARACTER == name_char)
    matches_name = os_markup.assign(**{"CHARACTER_" + name_char: positive_match})
    used_names = ["CHARACTER_" + name_char] + used_names
    print(matches_name)

    for top_name in top_eight_list:
        if "CHARACTER_" + top_name not in used_names:
            positive_matches = 1 * (os_markup.CHARACTER == top_name)
            matches_name = matches_name.assign(**{"CHARACTER_" + top_name: positive_matches})
            used_names = used_names + [top_name]

    match_word_counts = matches_thresh.groupby(
        'ORIGINAL_SCRIPT_WORD_INDEX'
    ).aggregate({
        name: numpy.sum for name in threshname
    })

    match_word_counts = match_word_counts.reindex(
        matches_name.index,
        fill_value=0
    )

    match_word_words = matches_thresh.groupby(
        'ORIGINAL_SCRIPT_WORD_INDEX'
    ).aggregate({
        'ORIGINAL_SCRIPT_WORD': numpy.max,
    })

    match_word_counts = match_word_counts.join(match_word_words)

    match_count = (match_word_counts.join(matches_name))

    match_count.to_csv(output)

def _format_data_sentiment_only(args):
    fin = args.s
    fout = args.o

    markup_script = search.load_markup_script(fin)
    markup_script = markup_script[1:]
    list_script = [[i] + r for i, r in enumerate(markup_script)]

    csv_script = pd.DataFrame(list_script)
    csv_script.columns = ['ORIGINAL_SCRIPT_INDEX',
       'LOWERCASE',
       'SPACY_ORTH_ID',
       'SCENE',
       'CHARACTER']

    bing_count = [bing.lex_count(j[1]) for j in list_script]
    bing_sentiment_keys = ['NEGATIVE', 'POSITIVE']
    bing_count = project_sentiment_keys_shortform(bing_count, bing_sentiment_keys)
    bing_DF = pd.DataFrame(bing_count)

    bing_DF['ORIGINAL_SCRIPT_INDEX'] = csv_script['ORIGINAL_SCRIPT_INDEX']
    out = pd.merge(csv_script, bing_DF, on='ORIGINAL_SCRIPT_INDEX')

    if emolex:
        emo_count = [emolex.lex_count(j[1]) for j in list_script]
        emo_sentiment_keys = ['ANTICIPATION', 'ANGER', 'TRUST', 'SADNESS','DISGUST',
                          'SURPRISE', 'FEAR', 'JOY', 'NEGATIVE', 'POSITIVE']
        emo_count = project_sentiment_keys_shortform(emo_count, emo_sentiment_keys)
        emo_DF = pd.DataFrame(emo_count)
        emo_DF['ORIGINAL_SCRIPT_INDEX'] = csv_script['ORIGINAL_SCRIPT_INDEX']
        out = pd.merge(out, emo_DF, on='ORIGINAL_SCRIPT_INDEX')

    if liwc:
        liwc_count = [liwc.lex_count(j[1]) for j in list_script]

        liwc_sentiment_keys = ['POSEMO', 'NEGEMO']
        liwc_sent_count = project_sentiment_keys_shortform(liwc_count, liwc_sentiment_keys)
        liwc_sent_DF = pd.DataFrame(liwc_sent_count)
        liwc_sent_DF['ORIGINAL_SCRIPT_INDEX'] = csv_script['ORIGINAL_SCRIPT_INDEX']
        out = pd.merge(out, liwc_sent_DF, on='ORIGINAL_SCRIPT_INDEX')

        liwc_other_keys = set(k for ct in liwc_count for k in ct.keys())
        liwc_other_keys -= set(['POSEMO', 'NEGEMO']) #already used these
        liwc_other_count = project_sentiment_keys_shortform(liwc_count, liwc_other_keys)
        liwc_other_DF = pd.DataFrame(liwc_other_count)
        liwc_other_DF['ORIGINAL_SCRIPT_INDEX'] = csv_script['ORIGINAL_SCRIPT_INDEX']
        out = pd.merge(out, liwc_other_DF, on='ORIGINAL_SCRIPT_INDEX')

    out.to_csv(fout + '.csv', index=False)

# -----------------------------------------------------------------------------
# SCRIPT
# ------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process fanworks scraped from Archive of Our Own.')
    subparsers = parser.add_subparsers(help='scrape, clean, getmeta, search, format, or vis')

    #sub-parsers
    scrape_parser = subparsers.add_parser('scrape', help='find and scrape fanfiction works from Archive of Our Own')
    group = scrape_parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--search', action='store', help="search term to search for a tag to scrape")
    group.add_argument('-t', '--tag', action='store', help="the tag to be scraped")
    group.add_argument('-u', '--url', action='store', help="the full URL of first page to be scraped")
    scrape_parser.add_argument('-o', '--out', action='store', default=os.path.join('.','scraped-html'), help="target directory for scraped html files")
    scrape_parser.add_argument('-p', '--startpage', action='store', default=1, type=int, help="page on which to begin downloading (to resume a previous job)")
    scrape_parser.set_defaults(func=scrape)

    clean_parser = subparsers.add_parser('clean', help='takes a directory of html files and yields a new directory of text files')
    clean_parser.add_argument('input', action='store', help='directory of input html files to clean')
    clean_parser.add_argument('-o', '--output', action='store', default='plain-text', help='target directory for output txt files')
    clean_parser.set_defaults(func=convert_dir)

    meta_parser = subparsers.add_parser('getmeta', help='takes a directory of html files and yields a csv file containing metadata')
    meta_parser.add_argument('input', action='store', help='directory of input html files to process')
    meta_parser.add_argument('-o', '--output', action='store', default='fan-meta', help='filename for metadata csv file')
    meta_parser.set_defaults(func=collect_meta)

    validate_parser = subparsers.add_parser('validate', help='validate script markup')
    validate_parser.add_argument('script', action='store', help='filename for markup version of script')
    validate_parser.set_defaults(func=search.validate_cmd)

    # Search for reuse
    search_parser = subparsers.add_parser('search', help='compare fanworks with the original script')
    search_parser.add_argument('fan_works', action='store', help='directory of fanwork text files')
    search_parser.add_argument('script', action='store', help='filename for markup version of script')
    search_parser.add_argument('-n', '--num-works', default=-1, type=int, help="number of works to search (for subsampling)")
    search_parser.set_defaults(func=search.analyze)

    # Aggregate word-level counts
    data_parser = subparsers.add_parser('format', help='takes a script and outputs a csv with senitment information for each word formatted for javascript visualization')
    data_parser.add_argument('matches', action='store', help='filename for search output')
    data_parser.add_argument('script', action='store', help='filename for markup version of script')
    data_parser.add_argument('-o', '--output', action='store', default='js-data.csv', help='filename for csv output file of data formatted for visualization')
    data_parser.set_defaults(func=format_data)

    # Generate visualizaiton
    vis_parser = subparsers.add_parser('vis',
                                       help='takes a formatted data csv '
                                            'and generates an embeddable '
                                            'visualization')
    vis_parser.add_argument('input', action='store',
                            help='the data csv to use for generating the '
                                 'visualization')
    vis_parser.add_argument('-s', '--static', action='store_true',
                            default=False,
                            help="save a full html file")
    vis_parser.add_argument('-o', '--output', action='store',
                            default='reuse.html',
                            help="output filename")
    vis_parser.add_argument('-w', '--words-per-chunk', type=int, default=140,
                            help='number of words per script segment')
    vis_parser.set_defaults(func=vis.save_plot)

    # handle args
    args = parser.parse_args()

    # call function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
