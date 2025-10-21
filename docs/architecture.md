flowchart TD
    %% User Layer
    subgraph User
        U1[User Uploads Documents]
        U2[User Queries / Q&A]
        U3[Dashboard for Monitoring]
    end

    %% Document Ingestion Layer
    subgraph Ingestion & Processing
        A1[MIME Detection & Parsing]
        A2[Chunking]
        A3[Chunk Enrichment (Summary, Keywords, Questions, Table)]
        A4[Logic Extraction (Claims, Relations, Assumptions, Constraints)]
        A5[Vectorization → Vector DB]
        A6[Metadata Storage → Knowledge DB]
    end

    %% Knowledge Consolidation Layer
    subgraph Knowledge_Consolidation
        B1[Document-Level Knowledge Extraction]
        B2[Subject-Level Knowledge Aggregation]
        B3[Clustering Agent (Topic Drift Monitoring)]
    end

    %% Agentic Q&A Layer
    subgraph QA_Agent
        C1[Retrieve from Vector DB (Intra-Doc)]
        C2[Retrieve from Knowledge DB (Inter-Subject)]
        C3[Reasoning with Logic Extraction]
        C4[Answer + Reasoning + Evidence]
    end

    %% MLOps & Monitoring Layer
    subgraph MLOps_Monitoring
        D1[Pipeline Monitoring]
        D2[Metrics: ingestion, logic extraction, clustering, retrieval]
        D3[CI/CD: Deployment, LLM Versioning, Batch & Online Inference]
        D4[Feedback Loop → Retraining / Knowledge Update]
    end

    %% Connections
    U1 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
    A4 --> A6
    A5 --> C1
    A6 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> C4
    U2 --> C4
    U3 --> D2
    D2 --> D4
    B1 --> B2
    B2 --> B3
    B2 --> C2
    B3 --> D2
    D4 --> A3
    D4 --> A4
