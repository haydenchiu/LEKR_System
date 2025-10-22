"""
Pydantic models for document enrichment.

Defines the data structures used for chunk enrichment including
summaries, keywords, questions, and table data.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkEnrichment(BaseModel):
    """Structured enrichment for a document chunk."""
    
    summary: str = Field(
        description="A concise 1-2 sentence summary of the chunk."
    )
    
    keywords: List[str] = Field(
        description="A list of 5-7 key topics or entities mentioned."
    )
    
    hypothetical_questions: List[str] = Field(
        description="A list of 3-5 questions this chunk could answer."
    )
    
    table_summary: Optional[str] = Field(
        description="If the chunk is a table, a natural language summary of its key insights.",
        default=None
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            # Add any custom encoders if needed
        }
        
    def model_dump(self, **kwargs):
        """Return model as dictionary."""
        return super().model_dump(**kwargs)
    
    def model_dump_json(self, **kwargs):
        """Return model as JSON string."""
        return super().model_dump_json(**kwargs)
