#!/bin/bash
# Reproduce results then text of the paper 
scriptDir="$(dirname "$0")"

# Make sure tlmgr (texlive manager) is installed and initialized
[[ "$(which tlmgr)" == "" ]] && echo 'tlmgr is not available; insall texlive and rerun'
[[ "$(which ~/.tlpkg)" == "" ]] && tlmgr init-usertree

texname=HAFiscal
output_directory='LaTeX'

pwd

# Compile LaTeX files in root directory
for file in "$texname" "$texname"-NoAppendix-NoTOC "$texname"-Slides; do
    if [[ -e "$file.tex" ]]; then
	echo '' ; echo "Compiling $file" ; echo ''
	#    dep="pwd ; texliveonfly $file"
	#    echo dep="$dep"
	#    eval "$dep"
	cmd="pdflatex -halt-on-error -output-directory=$output_directory $file"
	echo "$cmd"
	eval "$cmd"
	eval "$cmd > /dev/null" # Hide second output to reduce clutter
	bibtex $output_directory/"$file"
	eval "$cmd" # Hide third output to reduce clutter
	eval "$cmd > /dev/null" 
	echo '' ; echo "Compiled $file" ; echo ''
    fi
done

# # Compile All-Figures and All-Tables
# for type in Figures Tables; do
#     # dep="texliveonfly $type/All-$type"
#     # echo "pwd ; $dep"
#     # eval "$dep"
#     cmd="pdflatex -halt-on-error -output-directory=$output_directory $type/All-$type"
#     echo "$cmd" ; eval "$cmd"
#     # If there is a .bib file, make the references
#     [[ -e "../$output_directory/$type/All-$type.aux" ]] && bibtex "$type/All-$type.bib" && eval "$cmd" && eval "$cmd" 
#     mv -f "$output_directory/All-$type.pdf" "$type"  # Move from the LaTeX output directory to the destination
# done

# # All the appendices can be compiled as standalone documents (they are "subfiles")
# # Make a list of all the appendices, put the list in the file /tmp/appendices
# find ./Appendices -name '*.tex' ! -name '*econtexRoot*' ! -name '*econtexPath*' -maxdepth 1 -exec echo {} \; > /tmp/appendices

# # For each appendix process it by pdflatex
# # If it contains a standalone bibliography, process that
# # Then rerun pdflatex to complete the processing and move the resulting pdf file

# while read appendixName; do
#     filename=$(basename ${appendixName%.*}) # Strip the path and the ".tex"
# #    dep="texliveonfly $filename"
# #    echo dep="$dep"
# #    eval "$dep"
#     cmd="pdflatex -halt-on-error                 --output-directory=$output_directory $appendixName"
#     echo "$cmd"
#     eval "$cmd"
#     if grep -q 'bibliography{' "$appendixName"; then # it has a bibliography
# 	bibtex $output_directory/$filename 
# 	eval "$cmd" 
#     fi
#     eval "$cmd"
#     cmd="mv $output_directory/$filename.pdf Appendices"
#     echo "$cmd"
#     eval "$cmd"
# done < /tmp/appendices

# [[ -e "$texname".pdf ]] && rm -f "$texname".pdf

# echo '' 

# if [[ -e "$output_directory/$texname.pdf" ]]; then
#     echo "Paper has been compiled to $output_directory/$texname.pdf"
#     echo "and copied to ./$texname.pdf"
#     cp "$output_directory/$texname.pdf" "./$texname.pdf"
# else
#     echo "Something went wrong and the paper is not in $output_directory/$texname.pdf"
# fi

# echo ''

