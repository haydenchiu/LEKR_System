"""
Pydantic models for knowledge consolidation.

This module defines the data models for representing consolidated knowledge
at document and subject levels, including key concepts, relations, and storage.
"""

from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class KeyConcept(BaseModel):
    """Represents a key concept extracted from document chunks."""
    
    concept_id: str = Field(description="Unique identifier for the concept")
    name: str = Field(description="Name or title of the concept")
    description: str = Field(description="Detailed description of the concept")
    category: str = Field(description="Category or domain of the concept")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the concept")
    source_chunks: List[str] = Field(description="IDs of source chunks that contributed to this concept")
    keywords: List[str] = Field(default_factory=list, description="Keywords associated with the concept")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class KnowledgeRelation(BaseModel):
    """Represents a relationship between concepts."""
    
    relation_id: str = Field(description="Unique identifier for the relation")
    source_concept: str = Field(description="ID of the source concept")
    target_concept: str = Field(description="ID of the target concept")
    relation_type: str = Field(description="Type of relationship (e.g., 'causal', 'hierarchical', 'temporal')")
    strength: float = Field(ge=0.0, le=1.0, description="Strength of the relationship")
    description: Optional[str] = Field(default=None, description="Description of the relationship")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting this relationship")
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class DocumentKnowledge(BaseModel):
    """Represents consolidated knowledge at the document level."""
    
    document_id: str = Field(description="Unique identifier for the document")
    title: str = Field(description="Title of the document")
    summary: str = Field(description="High-level summary of the document")
    key_concepts: List[KeyConcept] = Field(default_factory=list, description="Key concepts extracted from the document")
    knowledge_relations: List[KnowledgeRelation] = Field(default_factory=list, description="Relationships between concepts")
    main_themes: List[str] = Field(default_factory=list, description="Main themes or topics in the document")
    knowledge_graph: Dict[str, Any] = Field(default_factory=dict, description="Graph representation of knowledge")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score of the consolidated knowledge")
    consolidation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the consolidation process")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    def get_concept_by_id(self, concept_id: str) -> Optional[KeyConcept]:
        """Get a concept by its ID."""
        for concept in self.key_concepts:
            if concept.concept_id == concept_id:
                return concept
        return None
    
    def get_relations_for_concept(self, concept_id: str) -> List[KnowledgeRelation]:
        """Get all relations involving a specific concept."""
        relations = []
        for relation in self.knowledge_relations:
            if relation.source_concept == concept_id or relation.target_concept == concept_id:
                relations.append(relation)
        return relations
    
    def get_concept_network(self, concept_id: str) -> Dict[str, Any]:
        """Get the network of concepts connected to a specific concept."""
        network = {
            "concept": self.get_concept_by_id(concept_id),
            "incoming_relations": [],
            "outgoing_relations": []
        }
        
        for relation in self.knowledge_relations:
            if relation.target_concept == concept_id:
                network["incoming_relations"].append(relation)
            elif relation.source_concept == concept_id:
                network["outgoing_relations"].append(relation)
        
        return network


class SubjectKnowledge(BaseModel):
    """Represents consolidated knowledge at the subject level."""
    
    subject_id: str = Field(description="Unique identifier for the subject")
    name: str = Field(description="Name of the subject")
    description: str = Field(description="Description of the subject")
    core_concepts: List[KeyConcept] = Field(default_factory=list, description="Core concepts for the subject")
    knowledge_relations: List[KnowledgeRelation] = Field(default_factory=list, description="Relationships between concepts")
    document_sources: List[str] = Field(default_factory=list, description="Document IDs that contributed to this subject knowledge")
    knowledge_hierarchy: Dict[str, Any] = Field(default_factory=dict, description="Hierarchical structure of knowledge")
    expertise_level: str = Field(default="intermediate", description="Level of expertise required (beginner, intermediate, advanced)")
    domain_tags: List[str] = Field(default_factory=list, description="Tags identifying the domain")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score of the subject knowledge")
    consolidation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the consolidation process")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    def get_concept_by_id(self, concept_id: str) -> Optional[KeyConcept]:
        """Get a concept by its ID."""
        for concept in self.core_concepts:
            if concept.concept_id == concept_id:
                return concept
        return None
    
    def get_related_concepts(self, concept_id: str, max_depth: int = 2) -> Set[str]:
        """Get concepts related to a specific concept within a certain depth."""
        related = set()
        to_explore = [(concept_id, 0)]
        
        while to_explore:
            current_id, depth = to_explore.pop(0)
            if depth >= max_depth:
                continue
                
            for relation in self.knowledge_relations:
                if relation.source_concept == current_id and relation.target_concept not in related:
                    related.add(relation.target_concept)
                    to_explore.append((relation.target_concept, depth + 1))
                elif relation.target_concept == current_id and relation.source_concept not in related:
                    related.add(relation.source_concept)
                    to_explore.append((relation.source_concept, depth + 1))
        
        return related


class KnowledgeStorage(BaseModel):
    """Represents storage information for knowledge."""
    
    storage_id: str = Field(description="Unique identifier for the storage entry")
    knowledge_type: str = Field(description="Type of knowledge (document or subject)")
    knowledge_id: str = Field(description="ID of the stored knowledge")
    storage_location: str = Field(description="Location where the knowledge is stored")
    storage_format: str = Field(description="Format of the stored knowledge (json, vector, etc.)")
    embedding_vector: Optional[List[float]] = Field(default=None, description="Vector embedding for similarity search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = Field(default=None, description="Last time this knowledge was accessed")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    def update_access_time(self):
        """Update the last accessed time."""
        self.last_accessed = datetime.now()
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information as a dictionary."""
        return {
            "storage_id": self.storage_id,
            "knowledge_type": self.knowledge_type,
            "knowledge_id": self.knowledge_id,
            "storage_location": self.storage_location,
            "storage_format": self.storage_format,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata
        }
