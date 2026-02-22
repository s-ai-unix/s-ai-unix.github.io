#!/bin/bash
conda run -n base python scripts/generate_riemann_plots.py
conda run -n base python scripts/generate_riemann_cover.py
