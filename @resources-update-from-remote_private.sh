#!/bin/bash
# Pull down the latest @resources and replace the existing one with it
/Volumes/Sync/GitHub/econ-ark/econ-ark-tools/@resources/bash/@resources-update-from-remote.sh "$(realpath $(dirname $0))" dryrun

