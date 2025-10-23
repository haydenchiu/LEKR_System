"""
Pydantic models for logic extraction results.

This module defines the data structures used to represent extracted logical and causal
relationships from document chunks, including claims, logical relations, assumptions,
constraints, and open questions.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class Claim(BaseModel):
    """Atomic logical or causal statement."""
    
    id: str = Field(..., description="Unique identifier for the claim.")
    statement: str = Field(..., description="The textual content of the claim.")
    type: Literal["factual", "inferential", "speculative", "assumptive"] = Field(
        ..., description="Type of claim based on its nature."
    )
    confidence: float = Field(
        0.5, 
        ge=0.0, 
        le=1.0, 
        description="Confidence level for the claim (0.0 to 1.0)."
    )
    derived_from: Optional[List[str]] = Field(
        default=None, 
        description="List of claim IDs that this claim is derived from."
    )
    
    model_config = ConfigDict(
        json_encoders={
            # Add any custom encoders if needed
        }
    )
    
    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)


class LogicalRelation(BaseModel):
    """Causal or inferential connection between claims."""
    
    premise: str = Field(..., description="Claim ID serving as the premise.")
    conclusion: str = Field(..., description="Claim ID serving as the conclusion.")
    relation_type: Literal[
        "causal", "inferential", "correlative", "contradictory", "supportive"
    ] = Field(..., description="Type of logical relationship.")
    certainty: float = Field(
        0.5, 
        ge=0.0, 
        le=1.0, 
        description="Certainty level for the relationship (0.0 to 1.0)."
    )
    
    model_config = ConfigDict(
        json_encoders={
            # Add any custom encoders if needed
        }
    )
    
    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)


class LogicExtractionSchemaLiteChunk(BaseModel):
    """Lightweight schema for representing logical and causal structure of a document chunk."""
    
    chunk_id: str = Field(..., description="Unique identifier of the chunk.")
    claims: List[Claim] = Field(
        default_factory=list, 
        description="List of extracted claims from the chunk."
    )
    logical_relations: List[LogicalRelation] = Field(
        default_factory=list, 
        description="List of logical relationships between claims."
    )
    assumptions: Optional[List[str]] = Field(
        default_factory=list, 
        description="Unstated but implied premises."
    )
    constraints: Optional[List[str]] = Field(
        default_factory=list, 
        description="Contextual or technical constraints."
    )
    open_questions: Optional[List[str]] = Field(
        default_factory=list, 
        description="Questions or uncertainties raised."
    )
    
    model_config = ConfigDict(
        json_encoders={
            # Add any custom encoders if needed
        }
    )
    
    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)
    
    def get_claim_by_id(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by its ID."""
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        return None
    
    def get_relations_by_premise(self, premise_id: str) -> List[LogicalRelation]:
        """Get all relations where the given claim ID is the premise."""
        return [rel for rel in self.logical_relations if rel.premise == premise_id]
    
    def get_relations_by_conclusion(self, conclusion_id: str) -> List[LogicalRelation]:
        """Get all relations where the given claim ID is the conclusion."""
        return [rel for rel in self.logical_relations if rel.conclusion == conclusion_id]
    
    def get_claim_network(self) -> dict:
        """Get the network structure of claims and their relationships."""
        network = {}
        for claim in self.claims:
            network[claim.id] = {
                "claim": claim,
                "premises": self.get_relations_by_premise(claim.id),
                "conclusions": self.get_relations_by_conclusion(claim.id)
            }
        return network
    
    def validate_claim_references(self) -> List[str]:
        """Validate that all claim references in relations exist."""
        claim_ids = {claim.id for claim in self.claims}
        errors = []
        
        for relation in self.logical_relations:
            if relation.premise not in claim_ids:
                errors.append(f"Premise claim ID '{relation.premise}' not found in claims")
            if relation.conclusion not in claim_ids:
                errors.append(f"Conclusion claim ID '{relation.conclusion}' not found in claims")
        
        return errors
