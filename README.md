The Archive of Our Own script ao3.py can be used to scrape and analyze fanworks and prepare the results for visualization in JavaScript.
A markup version of the script of the orginal work is required for searching for n-gram matches in the fanworks.

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

