
# coding: utf-8

import re
import os
import json
import csv
from bs4 import BeautifulSoup

# "AOOO_UNSPECIFIED" specifically means that An Archive of Our Own
# did not specify the value in their metadata fields.

def select_text(soup_node, selector):
    sel = soup_node.select(selector)
    return sel[0].get_text().strip() if sel else 'AOOO_UNSPECIFIED'

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

def collect_meta(in_dir, out_file):
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

if __name__ == '__main__':
    collect_meta('scraped-html', 'fan-meta')

