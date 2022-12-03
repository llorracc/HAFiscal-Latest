#!/bin/bash
# Should contain code which, when run, executes all of the publicly runnable code that produces any result in the paper
# If there are proprietary data, the commands to produce results from that data should be included
# even if the data are not there

if [ "$#" -gt 1 ]; then
    echo "usage:   "$toolRoot/${0##*/}" [Public | Private] "
    echo "example: "$toolRoot/${0##*/}" Public"
    exit 1
fi

if [[ "$#" -eq 0 ]]; then # default if run with no argument is to do the public version
    PubOrPri="Public"
else
    PubOrPri="$1" # If an argument exists it should be either "Public" or "Private"
fi

if [[ ( "$PubOrPri" != *"rivate" ) && ( "$PubOrPri" != *"ublic" ) ]]; then
    echo 'Argument should be one of two words: '"'Public'"' or '"'Private'"''
    exit 1
fi

if [[ "$PubOrPri" == *"rivate" ]]; then
    /bin/bash do_all_code_private.sh
else
    /bin/bash do_all_code_public.sh
fi

