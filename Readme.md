# Phase-Field-Crystal

A modular 2D Phase Field Crystal (PFC) simulation framework written in Python.

## Features

* Spectral semi-implicit PFC solver
* Hexagonal lattice
* Square lattice
* Triangular lattice
* Defect density analysis
* Voronoi tessellation
* Psi6 orientational order parameter
* Grain boundary detection
* Structure factor analysis
* Elastic energy calculation
* Stress-strain curve calculation

## Project Structure

pfc_base.py
Core numerical infrastructure

pfc_pure.py
Pure material PFC solver

pfc_analysis.py
Defect and microstructure analysis

pfc_plot.py
Visualization tools

pfc_elastic.py
Elasticity calculations

pfc_io.py
Video recording and output

run_pure.py
Run standard PFC simulation

run_elastic.py
Run elastic constant calculation

config.py
User interface menu

## Example

Run a standard PFC simulation:

python run_pure.py

Run elastic analysis:

python run_elastic.py

## Author

Jinpeng Wang

Department of Material Engineering

Mitacs Intern @ McMaster University
