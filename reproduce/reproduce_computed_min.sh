#!/bin/bash

# Make sure the necessary requirements are available
source ./reproduce/reproduce_environment.sh

# Change directory to the location of the Python script
cd Code/HA-Models

# Create version file with '_min' for minimal reproduction
rm -f version
echo "_min" > version

# Run the Python script
python reproduce_min.py
