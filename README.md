# README.md
# HOGAT: Higher-Order Graph Attention Networks for Vulnerability Detection in Smart Contracts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of HOGAT, a novel graph-based deep learning model for detecting vulnerabilities in smart contracts, as described in the paper "HOGAT: Higher-Order Graph Attention Networks for Vulnerability Detection in Smart Contracts".

## Overview

Smart contracts are prone to vulnerabilities like reentrancy, timestamp dependency, integer overflow/underflow, and infinite loops. HOGAT uses multi-hop attention mechanisms in a unified contract graph to capture long-range dependencies, outperforming baselines with an average F1-score of 89.8% on datasets like ESC, VSC, and SolidiFI.

Key contributions:
- Higher-order graph attention for long-range dependency capture.
- Code normalization and specialized contract graph construction.
- Comprehensive evaluation across multiple vulnerabilities and platforms.

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for training, as per paper's RTX 2060 setup)

### Using pip
1. Clone the repository: