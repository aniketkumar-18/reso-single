"""Entry point — run with: uvicorn main:app --reload"""

from src.gateway.app import create_app

app = create_app()
