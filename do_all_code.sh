#!/bin/bash
# Should contain code which, when run, executes all of the publicly runnable code that produces any result in the paper
# If there are proprietary data, the commands to produce results from that data should be included

if [ $# -gt 1 ]; then
  echo "usage:   "$toolRoot/${0##*/}" [Public | Private] "
  echo "example: "$toolRoot/${0##*/}" Public"
  exit 1
fi

if [[ $# -eq 0 ]]; then # default if run with no argument is to do the public version
    PubOrPri=Public
else
    PubOrPri="$1" # If an argument exists it should be either "Public" or "Private"
    if [[ "PubOrPri" != "Public" ]]; then
	if [[ "PubOrPri" != "Private" ]]; then
	    echo 'Argument should be one of two words: '"'Public'"' or '"'Private'"''
	    exit 1
fi

if [[ "$PubOrPri" == "Private" ]]; then
    /bin/bash do_all_code_pri.sh
fi

# Now execute the 'public' code 
/bin/bash do_all_code_pub.sh

