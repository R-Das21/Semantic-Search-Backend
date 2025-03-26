#!/bin/bash
gunicorn -w 1 -b 0.0.0.0:10000 semantic_search_api:app
