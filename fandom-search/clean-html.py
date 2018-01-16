
# coding: utf-8

import re
import os
from bs4 import BeautifulSoup

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

def convert_dir(html_dir, out_dir):
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

if __name__ == '__main__':
    convert_dir('scraped-html', 'plaintext')

