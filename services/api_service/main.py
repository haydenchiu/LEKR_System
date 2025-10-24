#!/usr/bin/env python3
"""
LERK System - API Service
This service provides REST API endpoints for the LERK System.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import LERK modules
from ingest import DocumentIngestionOrchestrator, DEFAULT_CONFIG
from enrichment import DocumentEnricher, DEFAULT_CONFIG as ENRICHMENT_CONFIG
from logic_extractor import LogicExtractor, DEFAULT_CONFIG as LOGIC_CONFIG
from clustering import DocumentClusterer, DEFAULT_CONFIG as CLUSTERING_CONFIG
from consolidation import DocumentConsolidator, SubjectConsolidator, DEFAULT_CONFIG as CONSOLIDATION_CONFIG
from qa_agent import QAOrchestrator, DEFAULT_QA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    document_id: str
    message: str
    processing_status: str


class DocumentStatusResponse(BaseModel):
    """Response model for document status."""
    document_id: str
    status: str
    progress: float
    chunks_count: int
    enriched_chunks: int
    logic_chunks: int
    created_at: str
    updated_at: str


class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., description="The question to answer")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    answer_style: str = Field("concise", description="Answer style: concise, detailed, academic, conversational")


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    success: bool
    answer: str
    sources: List[str]
    confidence: float
    session_id: str
    processing_time: float


class ServiceStatusResponse(BaseModel):
    """Response model for service status."""
    service_name: str
    version: str
    status: str
    timestamp: str
    uptime: float


# Global service instances
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LERK API Service")
    
    try:
        # Initialize services
        services['ingestion'] = DocumentIngestionOrchestrator(DEFAULT_CONFIG)
        services['enricher'] = DocumentEnricher(ENRICHMENT_CONFIG)
        services['logic_extractor'] = LogicExtractor(LOGIC_CONFIG)
        services['clusterer'] = DocumentClusterer(CLUSTERING_CONFIG)
        services['document_consolidator'] = DocumentConsolidator(CONSOLIDATION_CONFIG)
        services['subject_consolidator'] = SubjectConsolidator(CONSOLIDATION_CONFIG)
        services['qa_orchestrator'] = QAOrchestrator(DEFAULT_QA_CONFIG)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LERK API Service")


# Create FastAPI app
app = FastAPI(
    title="LERK System API",
    description="Logic Extraction and Reasoning Knowledge System API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service start time for uptime calculation
service_start_time = time.time()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "LERK System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=ServiceStatusResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check service health
        uptime = time.time() - service_start_time
        
        return ServiceStatusResponse(
            service_name="lerk-api-service",
            version="1.0.0",
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_async: bool = Form(True)
):
    """Upload and process a document."""
    try:
        # Generate document ID
        document_id = f"doc_{int(time.time())}_{file.filename}"
        
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{document_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        if process_async:
            # Process in background
            background_tasks.add_task(process_document_background, str(file_path), document_id)
            
            return DocumentUploadResponse(
                success=True,
                document_id=document_id,
                message="Document uploaded successfully. Processing in background.",
                processing_status="processing"
            )
        else:
            # Process synchronously
            result = await process_document_sync(str(file_path), document_id)
            
            return DocumentUploadResponse(
                success=result['success'],
                document_id=document_id,
                message=result.get('message', 'Document processed'),
                processing_status="completed" if result['success'] else "failed"
            )
            
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Get document processing status."""
    try:
        # Check if document exists
        upload_dir = Path("data/uploads")
        document_files = list(upload_dir.glob(f"{document_id}_*"))
        
        if not document_files:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check processing status
        processed_dir = Path("data/processed")
        processed_files = list(processed_dir.glob(f"{document_id}_*"))
        
        if processed_files:
            # Document is processed
            with open(processed_files[0], 'r') as f:
                processed_data = json.load(f)
            
            return DocumentStatusResponse(
                document_id=document_id,
                status="completed",
                progress=100.0,
                chunks_count=processed_data['processing_stats']['total_chunks'],
                enriched_chunks=processed_data['processing_stats']['enriched_chunks'],
                logic_chunks=processed_data['processing_stats']['logic_chunks'],
                created_at=processed_data['processing_timestamp'],
                updated_at=processed_data['processing_timestamp']
            )
        else:
            # Document is still processing
            return DocumentStatusResponse(
                document_id=document_id,
                status="processing",
                progress=50.0,  # Estimate
                chunks_count=0,
                enriched_chunks=0,
                logic_chunks=0,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/questions/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the LERK system."""
    try:
        start_time = time.time()
        
        # Use QA orchestrator to answer question
        response = services['qa_orchestrator'].ask_question(
            question=request.question,
            session_id=request.session_id,
            answer_style=request.answer_style
        )
        
        processing_time = time.time() - start_time
        
        return QuestionResponse(
            success=True,
            answer=response['answer'],
            sources=response.get('sources', []),
            confidence=response.get('confidence', 0.8),
            session_id=response.get('session_id', ''),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all processed documents."""
    try:
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            return []
        
        documents = []
        for file_path in processed_dir.glob("*_processed.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            documents.append({
                'document_id': file_path.stem.replace('_processed', ''),
                'file_path': data['file_path'],
                'chunks_count': data['processing_stats']['total_chunks'],
                'enriched_chunks': data['processing_stats']['enriched_chunks'],
                'logic_chunks': data['processing_stats']['logic_chunks'],
                'processed_at': data['processing_timestamp']
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its processed data."""
    try:
        # Delete uploaded file
        upload_dir = Path("data/uploads")
        upload_files = list(upload_dir.glob(f"{document_id}_*"))
        for file_path in upload_files:
            file_path.unlink()
        
        # Delete processed file
        processed_dir = Path("data/processed")
        processed_files = list(processed_dir.glob(f"{document_id}_*"))
        for file_path in processed_files:
            file_path.unlink()
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_service_stats():
    """Get service statistics."""
    try:
        # Get statistics from services
        ingestion_stats = services['ingestion'].get_ingestion_stats()
        enrichment_stats = services['enricher'].get_enrichment_stats()
        logic_stats = services['logic_extractor'].get_extraction_stats()
        
        return {
            'ingestion': ingestion_stats,
            'enrichment': enrichment_stats,
            'logic_extraction': logic_stats,
            'service_uptime': time.time() - service_start_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get service stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_background(file_path: str, document_id: str):
    """Process document in background."""
    try:
        logger.info(f"Processing document {document_id} in background")
        
        # Process document
        result = await process_document_sync(file_path, document_id)
        
        if result['success']:
            logger.info(f"Document {document_id} processed successfully")
        else:
            logger.error(f"Document {document_id} processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")


async def process_document_sync(file_path: str, document_id: str) -> Dict[str, Any]:
    """Process document synchronously."""
    try:
        # Step 1: Document Ingestion
        ingestion_result = services['ingestion'].ingest_document(file_path)
        
        if not ingestion_result['success']:
            return {
                'success': False,
                'error': f"Ingestion failed: {ingestion_result['error']}"
            }
        
        chunks = ingestion_result['chunks']
        
        # Step 2: Chunk Enrichment
        enriched_chunks = []
        for chunk in chunks:
            try:
                enriched_chunk = await services['enricher'].enrich_chunk_async(chunk)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.warning(f"Failed to enrich chunk: {e}")
                enriched_chunks.append(chunk)
        
        # Step 3: Logic Extraction
        logic_chunks = []
        for chunk in enriched_chunks:
            try:
                logic_chunk = await services['logic_extractor'].extract_logic_async(chunk)
                logic_chunks.append(logic_chunk)
            except Exception as e:
                logger.warning(f"Failed to extract logic from chunk: {e}")
                logic_chunks.append(chunk)
        
        # Step 4: Save processed document
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = processed_dir / f"{document_id}_processed.json"
        processed_data = {
            'file_path': file_path,
            'processing_timestamp': datetime.utcnow().isoformat(),
            'chunks': [chunk.model_dump() for chunk in logic_chunks],
            'processing_stats': {
                'total_chunks': len(chunks),
                'enriched_chunks': len(enriched_chunks),
                'logic_chunks': len(logic_chunks)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'message': 'Document processed successfully',
            'chunks_count': len(chunks),
            'enriched_chunks': len(enriched_chunks),
            'logic_chunks': len(logic_chunks)
        }
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main entry point for the API service."""
    try:
        # Get configuration from environment
        host = os.getenv('API_HOST', '0.0.0.0')
        port = int(os.getenv('API_PORT', '8000'))
        workers = int(os.getenv('API_WORKERS', '4'))
        
        logger.info(f"Starting LERK API Service on {host}:{port}")
        
        # Run the service
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            reload=False
        )
        
    except Exception as e:
        logger.error(f"API service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
