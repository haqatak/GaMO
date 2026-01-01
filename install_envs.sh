#!/usr/bin/env bash

# =================================================================
# Project Environment Setup Script
# This script creates 3 separate conda environments for the pipeline.
# =================================================================

set -e

# Color settings for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting installation of project environments...${NC}"

# 1. Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Error: conda command not found. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

# 2. Create 3dgs environment
echo -e "\n${GREEN}[1/3] Creating '3dgs' environment...${NC}"
if conda env list | grep -q "3dgs"; then
    echo -e "${YELLOW}⚠️ Environment '3dgs' already exists, skipping.${NC}"
else
    conda env create -f env_3dgs.yml
fi

# 3. Create taming2 environment
echo -e "\n${GREEN}[2/3] Creating 'taming2' environment...${NC}"
if conda env list | grep -q "taming2"; then
    echo -e "${YELLOW}⚠️ Environment 'taming2' already exists, skipping.${NC}"
else
    conda env create -f env_taming.yml
fi

# 4. Create GaMO environment
echo -e "\n${GREEN}[3/3] Creating 'GaMO' environment...${NC}"
if conda env list | grep -q "GaMO"; then
    echo -e "${YELLOW}⚠️ Environment 'GaMO' already exists, skipping.${NC}"
else
    conda env create -f env_GaMO.yml
fi

echo -e "\n${GREEN}🎉 All environments installed successfully!${NC}"
echo "You can now run the main pipeline script."