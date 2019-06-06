The Archive of Our Own script ao3.py can be used to scrape and analyze 
fanworks and prepare the results for visualization in JavaScript.
A markup version of the script of the orginal work is required for 
searching for n-gram matches in the fanworks.

The basic workflow is below. This assumes you have a `scripts` folder, 
a `fanworks` folder, and a `results` folder, with a particular structure
that can be inferred from the example commands below. (Sorry, very busy!)
Take "sw-all" to be a stand-in for a folder of fan works, "sw-new-hope.txt"
to be a stand-in for a correctly formatted script, and "sw-new-hope" (without
the .txt) to be a stand-in for the results folder for the given movie.

A todo for this repo is to create options for where to save error and 
log files, and search results.

Another todo for this repo is to create more thorough documentation, 
especially of the script format, which is idiosyncratic but effective.

* Scrape AO3 (Ooops! Currently broken!)

        python ao3.py scrape \
            -t "Star Wars - All Media Types" \
            -o fanworks/sw-all/html

The scrape command will save log and error files; check to see that the
scrape went OK, and then move the (generically named) error file to
`fanworks/sw-all/sw-all-errors.txt`.

* Clean the HTML

        python ao3.py clean \
            fanworks/sw-all/html/ \
            -o fanworks/sw-all/plaintext/

The clean command will save an error file; check to see that the cleaning
process went OK, and then move the error file (this time in the root dir)
from `clean-html-errors.txt` to `sw-all-clean-errors.txt'

* Perform the reuse search

        python ao3.py search \
            fanworks/sw-all/ \
            scripts/sw-new-hope.txt

The search command will create sevaral (and in some case, many, even hundreds)
of separate CSV files. Each one contains the results for 500 fan works. They
will automatically be aggregated by the script at the end of the process, but
they are also saved here to ensure that if the search is interrupted, the 
results are still usable.

If the search completes without any errors, the final aggregated data will
be in a file with a date timestamp in YYYYMMDD format. It will be something 
like "match-6gram-20190604." Create a new folder `results/sw-all/20190604/`, 
and move all the CSV files into that folder.

* Aggregate the results over the script (i.e. "format" the results)

        python ao3.py format \
            results/sw-new-hope/20190604/match-6gram-20190604.csv \
            scripts/sw-new-hope.txt \
            -o results/sw-new-hope/fandom-data-new-hope.csv

* Create a Bokeh visualization of the aggregated results

        python ao3.py vis \
            results/sw-new-hope/fandom-data-new-hope.csv \
            -o results/sw-new-hope/new_hope_reuse.html


This is not a perfect workflow and needs to be tidied up in several ways. I 
will get around to that someday.

```
usage: ao3.py [-h] {scrape,clean,getmeta,search,matrix,format} ...

process fanworks scraped from Archive of Our Own.

positional arguments:
  {scrape,clean,getmeta,search,matrix,format}
                        scrape, clean, getmeta, search, matrix, or format
    scrape              find and scrape fanfiction works from Archive of Our
                        Own
    clean               takes a directory of html files and yields a new
                        directory of text files
    getmeta             takes a directory of html files and yields a csv file
                        containing metadata
    search              compare fanworks with the original script
    matrix              deduplicates and builds matrix for best n-gram matches
    format              takes a script and outputs a csv with senitment
                        information for each word formatted for javascript
                        visualization

optional arguments:
  -h, --help            show this help message and exit
```
There are three scraping options for Archive of Our Own:
(1) Use the '-s' option to provide a search term and see a list of possible tags.
(2) Use the '-t' option to scrape fanworks from a tag.
(3) Use the '-u' option to scrape fanworks from a URL. The URL should be to the /works page,
	e.g. https://archiveofourown.org/tags/Rogue%20One:%20A%20Star%20Wars%20Story%20(2016)/works
```
usage: ao3.py scrape [-h] [-s SEARCH | -t TAG | -u URL] [-o OUT]
                     [-p STARTPAGE]

optional arguments:
  -h, --help            show this help message and exit
  -s SEARCH, --search SEARCH
                        search term to search for a tag to scrape
  -t TAG, --tag TAG     the tag to be scraped
  -u URL, --url URL     the full URL of first page to be scraped
  -o OUT, --out OUT     target directory for scraped html files
  -p STARTPAGE, --startpage STARTPAGE
                        page on which to begin downloading (to resume a
                        previous job)
``` 
Clean and convert the scraped html files into plain text files.
```
usage: ao3.py clean [-h] [-o O] i

positional arguments:
  i           directory of input html files to clean

optional arguments:
  -h, --help  show this help message and exit
  -o O        target directory for output txt files
```
Extract Archive of Our Own metadata from the scraped html files.
```
usage: ao3.py getmeta [-h] [-o O] i

positional arguments:
  i           directory of input html files to process

optional arguments:
  -h, --help  show this help message and exit
  -o O        filename for metadata csv file
```
The search process compares fanworks with the original work script and is based on 6-gram matches.
```
usage: ao3.py search [-h] d s

positional arguments:
  d           directory of fanwork text files
  s           filename for markup version of script

optional arguments:
  -h, --help  show this help message and exits
```
The n-gram search results can be used to create a matrix.
```
usage: ao3.py matrix [-h] [-n N] i m

positional arguments:
  i           input csv file
  m           fandom/movie name for output file prefix

optional arguments:
  -h, --help  show this help message and exit
  -n N        n-gram size, default is 6-grams
```
The n-gram search results can be prepared for JavaScript visualization.
```
usage: ao3.py format [-h] [-o O] s

positional arguments:
  s           filename for markup version of script

optional arguments:
  -h, --help  show this help message and exit
  -o O        filename for csv output file of data formatted for visualization
s```

