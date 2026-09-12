# Damped Harmonic Oscillator Analysis

A Python-based data processing pipeline that analyzes damped pendulum oscillations to examine fluid drag forces across varying cross-sectional areas. 

This repository contains the analysis scripts, raw tracking data, and statistical visualization tools developed for an experimental physics investigation on energy dissipation regimes (laminar vs. turbulent drag).

---

## Project Overview

Ideal simple harmonic motion assumes zero energy loss, but real-world oscillations experience exponential damping due to aerodynamic drag. This project automates the extraction and modeling of the exponential damping coefficient ($b$) from time-series amplitude data.

- **Objective:** Determine how altering the cross-sectional area ($A$) of a pendulum bob affects its damping coefficient ($b$).
- **Experimental Range:** Area varied from $0.0081\text{ m}^2$ to $0.0289\text{ m}^2$ with mass ($0.086\text{ kg}$) and release angle ($10^\circ$) strictly controlled.
- **Key Finding:** Power-law linearization yields $b \propto A^{1.035}$, confirming a directly proportional linear drag relationship dominated by laminar skin friction (Stokes' Drag) rather than quadratic pressure drag.

---

## Data Processing Pipeline

1. **Amplitude Envelope Extraction:** Linearizes exponential decay curves using natural logarithms ($\ln\vert{}x\vert{} = -bt + \ln A_0$) to calculate $b$ for each trial.
2. **Error Propagation & Quadrature:** Merges spatial/temporal camera tracking uncertainties (30 fps frame rate and pixel resolution) with the Standard Error of the Mean (SEM) using quadrature error propagation:
   $$\text{Uncertainty} = \sqrt{(\text{SEM})^2 + (U_i)^2}$$
3. **Power-Law Scaling:** Fits $b = kA^n$ via log-log regression ($\ln b = n \ln A + \ln k$) to derive the empirical power-law exponent ($n$).
4. **Model Validation:** Computes residual distributions to verify model adequacy and evaluate systematic errors (e.g., pivot mechanical friction).

---

## Repository Structure

```text
├── data/                  # Raw position-time CSV files extracted from Tracker
├── scripts/               # Python scripts for regression, error propagation, and plotting
├── plots/                 # Exported decay envelopes, log-log fits, and residual plots
├── README.md              # Project documentation
