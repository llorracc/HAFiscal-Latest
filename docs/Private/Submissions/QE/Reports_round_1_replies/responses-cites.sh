#!/bin/bash

file=responses-cites-bib

# Find all cite commands and extract them
grep -oh '\\cite[a-zA-Z]*{[^}]*}' *.tex | \
# Replace all variations of \cite with simple \cite
sed 's/\\cite[a-zA-Z]*/\\cite/' | \
# Remove duplicates
sort -u | \
# Replace \cite with \nocite
sed 's/\\cite/\\nocite/' > responses-cites.tex

pdflatex responses-cites-bib
bibtex   responses-cites-bib
		 syslib="$(kpsewhich system.bib)"
		 addlib="$(kpsewhich $file-Add-Refs.bib)"
		 cmd="bibtool -x $file.aux -o $file.bib $syslib $addlib"
		 echo "$cmd"
		 eval "$cmd"
pdflatex responses-cites-bib
pdflatex responses-cites-bib


