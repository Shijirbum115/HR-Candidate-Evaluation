#!/bin/bash
source .env
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload