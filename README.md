# LEKR_System
Logic Extraction & Knowledge Reasoning System
Table of Contents

Project Overview

Architecture

Features

Installation

Usage

Data Handling

MLOps & Monitoring

Feedback & Iteration

License

Project Overview

This project aims to extract logical structures, causal relations, and key knowledge from documents, and consolidate them into document-level and subject-level knowledge databases. Users can query the system with a question and receive answers with reasoning paths and supporting evidence.

Goals:

Transform unstructured documents (PDF, HTML, Word, etc.) into enriched and logic-augmented chunks.

Build intra-document and inter-subject knowledge bases for reasoning and retrieval.

Enable agentic Q&A with logical explanations.

Support multi-user workflows with potential per-user namespaces.

Pain Points Addressed:

Difficulty in understanding complex logical flows in research papers or domain-specific documents.

Fragmented knowledge across documents and topics.

Limited ability to query documents with reasoning and causality.

Architecture

The system is organized into five layers:

User Layer:

Upload documents and query the system.

Dashboard for monitoring ingestion, clustering, and retrieval.

Document Ingestion & Processing:

MIME detection & parsing (supports HTML URLs, PDFs, DOCX).

Chunking, enrichment (summary, keywords, table summaries, hypothetical questions).

Logic extraction (claims, logical relations, assumptions, constraints, open questions).

Vectorization and metadata storage.

Knowledge Consolidation:

Document-level knowledge extraction from enriched logic chunks.

Subject-level aggregation across multiple documents.

Clustering agent to detect topic drift and maintain cluster coherence.

Agentic Q&A:

Retrieves relevant chunks and knowledge from vector DB and knowledge DB.

Performs reasoning with logical structures.

Returns answers with explanations and supporting evidence.

MLOps & Monitoring:

Pipeline monitoring, performance metrics, and alerts.

CI/CD with versioned LLM prompts and schemas.

Feedback loops for retraining, knowledge updates, and topic drift adjustments.

For a visual overview, see the Mermaid diagram in the docs/architecture.md file.

Features

Multi-format document ingestion with MIME detection and error handling.

Chunk enrichment and logic extraction via LLM.

Document-level and subject-level knowledge consolidation.

Vector database for retrieval and agentic reasoning.

Clustering agent for topic detection and drift monitoring.

Q&A agent supporting intra-document and inter-subject knowledge retrieval.

MLOps monitoring, logging, and CI/CD pipeline integration.

Installation

Clone the repository:

git clone https://github.com/yourusername/logic-extraction-system.git
cd logic-extraction-system


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Set up environment variables:

cp .env.example .env
# Edit .env with your API keys and database credentials


Start Jupyter/VSCode or Cursor for development.

Usage
Processing Files
from tools import process_file_tool

status = process_file_tool("data/research_paper.pdf")
print(status)

Querying with Q&A Agent
from tools import retriever_tool

response = retriever_tool("Explain the causal relationship between X and Y in the uploaded documents.")
print(response)

Adding Knowledge Consolidation
from knowledge_consolidation import consolidate_document_knowledge, consolidate_subject_knowledge

# Document-level
consolidate_document_knowledge(document_id="doc_123")

# Subject-level
consolidate_subject_knowledge(subject_id="subject_abc")

Data Handling

Input: PDF, HTML, DOCX (other types via MIME detection).

Chunks: Text or table-based, enriched with summary, keywords, and questions.

Logic Extraction: Claims, relations, assumptions, constraints, open questions.

Storage:

Vector DB: embeddings for retrieval.

Knowledge DB: structured metadata and consolidated knowledge.

MLOps & Monitoring

Pipeline Monitoring: ingestion, chunking, enrichment, logic extraction, clustering.

Metrics: chunk creation, LLM usage, retrieval quality, cluster drift.

CI/CD: Dockerized pipelines, automated deployment, LLM prompt versioning.

Feedback Loops: user corrections, monitoring metrics → retraining and knowledge updates.

Feedback & Iteration

Users can provide feedback on Q&A responses.

Metrics and logs are collected to detect:

Topic drift

Cluster reassignment needs

LLM output quality

Feedback triggers pipeline retraining and knowledge DB updates.

License

This project is licensed under the MIT License. See LICENSE
 for details.