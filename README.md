# fandom-search
An approximate nearest-neighbor search for text reuse.

### Scraping Works of Fanfiction from Archive of Our Own

Run the scrape-ao3 script either the URL or Tag options, as follows:

```
usage: scrape-ao3.py [-h] [-s SEARCH | -t TAG | -u URL] [-o OUTFILE]

Parse Fanfiction

optional arguments:
  -h, --help            show this help message and exit
  -s SEARCH, --search SEARCH
                        search term to search for a tag to scrape
  -t TAG, --tag TAG     the tag to be scraped
  -u URL, --url URL     the full URL of first page to be scraped
  -o OUTFILE, --outfile OUTFILE
                        target directory for scraped files
  ```
  
Do not use the --outfile option. The default target directory is scraped-html, and using a different output directory will be incompatible with the cleaning and processing scripts.

For example, to scrape the Star Wars Episode VII: The Force Awakens (2015) fanfiction words, use:
```
scrape-ao3.py -u http://archiveofourown.org/tags/Star%20Wars%20Episode%20VII:%20The%20Force%20Awakens%20(2015)/works
```

### Processing the HTML files

Run clean-html.py, which will output plain text versions of the fan works in a directory called 'plaintext'.
Next, run getmeta-html.py, which will create a csv file with metadata called 'fan-meta.csv'.

### Nearest Neighbor Search

Finally, run script-fan-comparison.py, which will create csv files with the results of the processing based on the scraped fanworks and the 'markup-script' in the 'original-scripts' directory. This script may take several hours to run, depending on the number of fanworks.
