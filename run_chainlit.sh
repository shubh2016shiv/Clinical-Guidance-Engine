#!/bin/bash
# Run Chainlit UI on port 8080 (to avoid conflict with Milvus Attu on port 8000)
chainlit run chatbot_ui/app.py --port 8080

