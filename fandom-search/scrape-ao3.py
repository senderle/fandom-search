import os
import requests
from bs4 import BeautifulSoup
import re
import argparse

if __name__ == "__main__":

    # command line interface
	parser = argparse.ArgumentParser(description='Parse Fanfiction')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-s', '--search', action='store', help="search term to search for a tag to scrape")
	group.add_argument('-t', '--tag', action='store', nargs=1, help="the tag to be scraped")
	group.add_argument('-u', '--url', action='store', nargs=1, help="the full URL of first page to be scraped")
	parser.add_argument('-o', '--outfile', action='store', nargs=1, default='scraped-html', help="target directory for scraped files")

	args = parser.parse_args()
	argv = vars(args)

	searchVal = argv['search']
	tagVal = argv['tag']
	headerVal = argv['url']
	targetVal = argv['outfile']

	tagList = ["initialize"]
	resultList = ["initialize"]

    # tag scraping option
	if searchVal:
		pp = 1
		safeSearch = searchVal.replace(' ', '+')
		# the alternative here is to scrape this page and use regex to filter the results:
		# http://archiveofourown.org/media/Movies/fandoms?
		# the canonical filter is used here because the "fandom" filter on the beta tag search is broken as of November 2017
		searchRef = "http://archiveofourown.org/tags/search?utf8=%E2%9C%93&query%5Bname%5D=" + safeSearch + "&query%5Btype%5D=&query%5Bcanonical%5D=true&page="
		print('\nTags:')

		while (len(tagList)) != 0:
			resPage1 = requests.get(searchRef + str(pp))
			resSoup1 = BeautifulSoup(resPage1.text, "lxml")
			tagList = resSoup1(attrs={'href': re.compile('^/tags/[^s]....[^?].*')})
			
			for x in tagList:
				print(x.string)
			
			pp += 1

    # fan work scraping options
	if headerVal or tagVal:
		end = 1 #pagination
		os.chdir(targetVal) #target directory

		while (len(resultList)) != 0:
			with open('log.txt', 'a') as f:
				f.write('\n\nPAGE ' + str(end) + '\n')
			print('page ' + str(end))
			
			if headerVal:
				header = headerVal[0]      
			
			if tagVal:
				modHeaderVal = tagVal[0].replace(' ', '%20')
				header = "http://archiveofourown.org/tags/" + modHeaderVal + "/works?page="       
			
			page1 = requests.get(header + "?page=" + str(end))
			soup1 = BeautifulSoup(page1.text, "lxml")
			resultList = soup1(attrs={'href': re.compile('^/works/[0-9]+[0-9]$')})

			with open("log.txt", "a") as f:
				f.write('Number of Works on Page ' + str(end) + ': ' + str((len(resultList))) + '\n')		
				f.write('Progress: \n') 
				
			for x in resultList:
				body = str(x).split('"')
				docID = str(body[1]).split('/')
				page2 = requests.get("http://archiveofourown.org/" + body[1] + "?view_adult=true&view_full_work=true")
				filename = str(docID[2]) + '.html'
				f = open(filename, 'w', encoding='UTF')
				f.write(str(page2.text))
				f.close()
				
				with open("log.txt", "a") as f:
					f.write("reached document " + str(docID[2]) + " on page " + str(end) + '\n')
				
			end += 1