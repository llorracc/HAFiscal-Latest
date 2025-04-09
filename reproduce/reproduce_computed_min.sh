#!/bin/bash

# Make sure the necessary requirements are available
source ./reproduce/reproduce_environment.sh

# Change directory to the location of the Python script
cd Code/HA-Models

# Create version file with '_min' for minimal reproduction
rm -f version
echo "_min" > version

# List of tables to manage
TABLES=(
    "Target_AggMPCX_LiquWealth/Figures/MPC_WealthQuartiles_Table.tex"
    "FromPandemicCode/Tables/CRRA2/Multiplier.tex"
    "FromPandemicCode/Tables/CRRA2/welfare6.tex"
    "FromPandemicCode/Tables/Splurge0/welfare6_SplurgeComp.tex"
    "FromPandemicCode/Tables/Splurge0/Multiplier_SplurgeComp.tex"
)

# Create backups of original tables
python3 ../reproduce/table_renamer.py backup "${TABLES[@]}"

# Run the minimal reproduction script
python reproduce_min.py

# Rename newly created tables to have _min suffix
python3 ../reproduce/table_renamer.py rename_min "${TABLES[@]}"

# Restore original tables
python3 ../reproduce/table_renamer.py restore "${TABLES[@]}"
