"""
LERK System - Services Module
This module contains dockerized microservices for the LERK System.
"""

__version__ = "1.0.0"
__author__ = "LERK System Team"

# Service modules
from .ingest_service import IngestService
from .api_service import APIService
from .worker_service import WorkerService
from .qa_service import QAService

__all__ = [
    "IngestService",
    "APIService", 
    "WorkerService",
    "QAService"
]
