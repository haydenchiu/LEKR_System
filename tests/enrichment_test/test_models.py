"""
Unit tests for enrichment models.

Tests the ChunkEnrichment Pydantic model and related functionality.
"""

import pytest
from pydantic import ValidationError

from enrichment.models import ChunkEnrichment


class TestChunkEnrichment:
    """Test cases for ChunkEnrichment model."""
    
    def test_init_default(self, sample_enrichment_data):
        """Test ChunkEnrichment initialization with default values."""
        enrichment = ChunkEnrichment(**sample_enrichment_data)
        
        assert enrichment.summary == sample_enrichment_data["summary"]
        assert enrichment.keywords == sample_enrichment_data["keywords"]
        assert enrichment.hypothetical_questions == sample_enrichment_data["hypothetical_questions"]
        assert enrichment.table_summary is None
    
    def test_init_with_table_summary(self, sample_table_enrichment_data):
        """Test ChunkEnrichment initialization with table summary."""
        enrichment = ChunkEnrichment(**sample_table_enrichment_data)
        
        assert enrichment.summary == sample_table_enrichment_data["summary"]
        assert enrichment.keywords == sample_table_enrichment_data["keywords"]
        assert enrichment.hypothetical_questions == sample_table_enrichment_data["hypothetical_questions"]
        assert enrichment.table_summary == sample_table_enrichment_data["table_summary"]
    
    def test_init_minimal_data(self):
        """Test ChunkEnrichment initialization with minimal required data."""
        data = {
            "summary": "Test summary",
            "keywords": ["keyword1", "keyword2"],
            "hypothetical_questions": ["Question 1?", "Question 2?"]
        }
        
        enrichment = ChunkEnrichment(**data)
        
        assert enrichment.summary == "Test summary"
        assert enrichment.keywords == ["keyword1", "keyword2"]
        assert enrichment.hypothetical_questions == ["Question 1?", "Question 2?"]
        assert enrichment.table_summary is None
    
    def test_validation_summary_required(self):
        """Test that summary is required."""
        data = {
            "keywords": ["keyword1"],
            "hypothetical_questions": ["Question 1?"]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChunkEnrichment(**data)
        
        assert "summary" in str(exc_info.value)
    
    def test_validation_keywords_required(self):
        """Test that keywords are required."""
        data = {
            "summary": "Test summary",
            "hypothetical_questions": ["Question 1?"]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChunkEnrichment(**data)
        
        assert "keywords" in str(exc_info.value)
    
    def test_validation_questions_required(self):
        """Test that hypothetical_questions are required."""
        data = {
            "summary": "Test summary",
            "keywords": ["keyword1"]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChunkEnrichment(**data)
        
        assert "hypothetical_questions" in str(exc_info.value)
    
    def test_validation_keywords_list(self):
        """Test that keywords must be a list."""
        data = {
            "summary": "Test summary",
            "keywords": "not a list",
            "hypothetical_questions": ["Question 1?"]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChunkEnrichment(**data)
    
    def test_validation_questions_list(self):
        """Test that hypothetical_questions must be a list."""
        data = {
            "summary": "Test summary",
            "keywords": ["keyword1"],
            "hypothetical_questions": "not a list"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChunkEnrichment(**data)
    
    def test_validation_table_summary_optional(self):
        """Test that table_summary is optional."""
        data = {
            "summary": "Test summary",
            "keywords": ["keyword1"],
            "hypothetical_questions": ["Question 1?"],
            "table_summary": "Table summary"
        }
        
        enrichment = ChunkEnrichment(**data)
        assert enrichment.table_summary == "Table summary"
    
    def test_model_dump(self, sample_enrichment_data):
        """Test model_dump method."""
        enrichment = ChunkEnrichment(**sample_enrichment_data)
        dumped = enrichment.model_dump()
        
        assert isinstance(dumped, dict)
        assert dumped["summary"] == sample_enrichment_data["summary"]
        assert dumped["keywords"] == sample_enrichment_data["keywords"]
        assert dumped["hypothetical_questions"] == sample_enrichment_data["hypothetical_questions"]
        assert dumped["table_summary"] is None
    
    def test_model_dump_with_table_summary(self, sample_table_enrichment_data):
        """Test model_dump method with table summary."""
        enrichment = ChunkEnrichment(**sample_table_enrichment_data)
        dumped = enrichment.model_dump()
        
        assert isinstance(dumped, dict)
        assert dumped["table_summary"] == sample_table_enrichment_data["table_summary"]
    
    def test_model_dump_json(self, sample_enrichment_data):
        """Test model_dump_json method."""
        enrichment = ChunkEnrichment(**sample_enrichment_data)
        json_str = enrichment.model_dump_json()
        
        assert isinstance(json_str, str)
        assert "summary" in json_str
        assert "keywords" in json_str
        assert "hypothetical_questions" in json_str
    
    def test_str_representation(self, sample_enrichment_data):
        """Test string representation of ChunkEnrichment."""
        enrichment = ChunkEnrichment(**sample_enrichment_data)
        str_repr = str(enrichment)
        
        # Pydantic v2 uses a different string representation format
        assert sample_enrichment_data["summary"] in str_repr
        assert "keywords=" in str_repr
    
    def test_equality(self, sample_enrichment_data):
        """Test equality of ChunkEnrichment instances."""
        enrichment1 = ChunkEnrichment(**sample_enrichment_data)
        enrichment2 = ChunkEnrichment(**sample_enrichment_data)
        
        assert enrichment1 == enrichment2
    
    def test_inequality(self, sample_enrichment_data):
        """Test inequality of ChunkEnrichment instances."""
        enrichment1 = ChunkEnrichment(**sample_enrichment_data)
        
        different_data = sample_enrichment_data.copy()
        different_data["summary"] = "Different summary"
        enrichment2 = ChunkEnrichment(**different_data)
        
        assert enrichment1 != enrichment2
    
    def test_hash(self, sample_enrichment_data):
        """Test hash of ChunkEnrichment instances."""
        enrichment1 = ChunkEnrichment(**sample_enrichment_data)
        enrichment2 = ChunkEnrichment(**sample_enrichment_data)
        
        # Pydantic models are not hashable by default
        # Test that they can be converted to hashable types
        enrichment1_dict = enrichment1.model_dump()
        enrichment2_dict = enrichment2.model_dump()
        
        # Convert to hashable tuples, handling nested lists
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            else:
                return obj
        
        enrichment1_tuple = make_hashable(enrichment1_dict)
        enrichment2_tuple = make_hashable(enrichment2_dict)
        
        assert hash(enrichment1_tuple) == hash(enrichment1_tuple)
        assert hash(enrichment1_tuple) == hash(enrichment2_tuple)
    
    def test_empty_keywords(self):
        """Test ChunkEnrichment with empty keywords list."""
        data = {
            "summary": "Test summary",
            "keywords": [],
            "hypothetical_questions": ["Question 1?"]
        }
        
        enrichment = ChunkEnrichment(**data)
        assert enrichment.keywords == []
    
    def test_empty_questions(self):
        """Test ChunkEnrichment with empty questions list."""
        data = {
            "summary": "Test summary",
            "keywords": ["keyword1"],
            "hypothetical_questions": []
        }
        
        enrichment = ChunkEnrichment(**data)
        assert enrichment.hypothetical_questions == []
    
    def test_long_summary(self):
        """Test ChunkEnrichment with long summary."""
        long_summary = "This is a very long summary that contains a lot of information about the document chunk and provides detailed insights into the content and its implications for the overall understanding of the subject matter."
        
        data = {
            "summary": long_summary,
            "keywords": ["keyword1"],
            "hypothetical_questions": ["Question 1?"]
        }
        
        enrichment = ChunkEnrichment(**data)
        assert enrichment.summary == long_summary
    
    def test_special_characters(self):
        """Test ChunkEnrichment with special characters."""
        data = {
            "summary": "Summary with special chars: @#$%^&*()",
            "keywords": ["keyword@1", "keyword#2", "keyword$3"],
            "hypothetical_questions": ["Question with @#$%^&*()?"]
        }
        
        enrichment = ChunkEnrichment(**data)
        assert enrichment.summary == "Summary with special chars: @#$%^&*()"
        assert "keyword@1" in enrichment.keywords
        assert "Question with @#$%^&*()?" in enrichment.hypothetical_questions
    
    def test_unicode_characters(self):
        """Test ChunkEnrichment with unicode characters."""
        data = {
            "summary": "Summary with unicode: 你好世界 🌍",
            "keywords": ["关键词1", "关键词2"],
            "hypothetical_questions": ["问题1？", "问题2？"]
        }
        
        enrichment = ChunkEnrichment(**data)
        assert "你好世界" in enrichment.summary
        assert "关键词1" in enrichment.keywords
        assert "问题1？" in enrichment.hypothetical_questions
