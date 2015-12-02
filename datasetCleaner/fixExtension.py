#!/usr/bin/env python

import os
from glob import glob
import sys

def fixExtension(directory):

	count = 0
	result = []
	for path, subdirs, files in os.walk(directory):
		for name in files:
			result.append(os.path.join(path, name))
	for page in result:
		print "Fixing file extension for webpage: ",page
		original = page
		filename, file_extension = os.path.splitext(page)
		if (file_extension == ".html"):
			page = page[:-4]
			txt_extension = page + "txt"
		else:
			txt_extension = page + ".txt"
		os.rename(original,txt_extension)
		count += 1
		ftxt = open(txt_extension, "rb")
		raw_data = ftxt.read()
		ftxt.close
		ftxtr = open(txt_extension, "wb")

		#print raw_data
		rawdata_soup = " ".join(line.strip() for line in raw_data.split("\n"))
		ftxtr.write(rawdata_soup)
		#print rawdata_soup
		ftxtr.close()
		print "Done fixing webpage: ", page
		print "Total webpages fixed so far: ", count
	print "Total pages fixed: %d" %count

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print >> sys.stderr, 'SYNTAX: fixExtension.py [directory]'
        sys.exit(-1)

    fixExtension(args[0])


