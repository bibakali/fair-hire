#!/bin/bash
cd "$(dirname "$0")"
source env/bin/activate
PYTHONPATH=$(pwd) streamlit run app/streamlit_app.py