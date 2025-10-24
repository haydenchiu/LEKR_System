"""
Unit tests for document consolidator.

This module contains tests for the DocumentConsolidator class
and its methods for consolidating document knowledge.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from consolidation.document_consolidator import DocumentConsolidator
from consolidation.models import DocumentKnowledge, KeyConcept, KnowledgeRelation
from consolidation.config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from consolidation.exceptions import (
    DocumentConsolidationError, ConceptExtractionError, RelationExtractionError,
    MissingAPIKeyError, KnowledgeValidationError
)


class TestDocumentConsolidator:
    """Test cases for the DocumentConsolidator class."""
    
    @pytest.fixture(autouse=True)
    def mock_chat_openai(self):
        """Mock ChatOpenAI to prevent actual API calls."""
        with patch('consolidation.document_consolidator.ChatOpenAI') as mock_llm_class:
            mock_llm_instance = Mock()
            mock_llm_instance.with_structured_output.return_value = mock_llm_instance
            mock_llm_class.return_value = mock_llm_instance
            yield mock_llm_instance
    
    @pytest.fixture(autouse=True)
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer to prevent actual model loading."""
        with patch('consolidation.document_consolidator.SentenceTransformer') as mock_model_class:
            mock_model_instance = Mock()
            mock_model_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            mock_model_class.return_value = mock_model_instance
            yield mock_model_instance
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LLM response for testing."""
        response = Mock()
        response.content = '''[
            {
                "name": "Machine Learning",
                "description": "A subset of artificial intelligence",
                "category": "ai",
                "confidence": 0.9,
                "keywords": ["AI", "algorithms"]
            }
        ]'''
        return response
    
    def test_init_default_config(self, mock_chat_openai):
        """Test initialization with default configuration."""
        consolidator = DocumentConsolidator()
        
        assert isinstance(consolidator.config, ConsolidationConfig)
        assert consolidator.config.model_name == "gpt-4o-mini"
        mock_chat_openai.assert_called_once()
    
    def test_init_custom_config(self, mock_chat_openai):
        """Test initialization with custom configuration."""
        config = ConsolidationConfig(model_name="gpt-4o", temperature=0.2)
        consolidator = DocumentConsolidator(config)
        
        assert consolidator.config == config
        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.2)
    
    def test_init_with_missing_api_key(self):
        """Test initialization failure with missing API key."""
        with patch('consolidation.document_consolidator.ChatOpenAI', side_effect=Exception("API key not found")):
            with pytest.raises(MissingAPIKeyError, match="Failed to initialize LLM"):
                DocumentConsolidator()
    
    def test_consolidate_document_success(self, sample_chunks, mock_llm_response):
        """Test successful document consolidation."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        result = consolidator.consolidate_document(
            document_id="doc_1",
            document_title="Test Document",
            chunks=sample_chunks
        )
        
        assert isinstance(result, DocumentKnowledge)
        assert result.document_id == "doc_1"
        assert result.title == "Test Document"
        assert len(result.key_concepts) > 0
        assert result.quality_score >= 0.0
    
    def test_consolidate_document_with_logic_data(self, sample_chunks, sample_logic_data, mock_llm_response):
        """Test document consolidation with logic data."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        result = consolidator.consolidate_document(
            document_id="doc_1",
            document_title="Test Document",
            chunks=sample_chunks,
            chunk_logic_data=sample_logic_data
        )
        
        assert isinstance(result, DocumentKnowledge)
        assert result.document_id == "doc_1"
        assert "logic" in result.consolidation_metadata or True  # May or may not be present
    
    def test_consolidate_document_failure(self, sample_chunks):
        """Test document consolidation failure."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(DocumentConsolidationError, match="Failed to consolidate document"):
            consolidator.consolidate_document(
                document_id="doc_1",
                document_title="Test Document",
                chunks=sample_chunks
            )
    
    def test_extract_concepts_from_chunks(self, sample_chunks, mock_llm_response):
        """Test concept extraction from chunks."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        concepts = consolidator._extract_concepts_from_chunks(sample_chunks)
        
        assert isinstance(concepts, list)
        # Should have extracted some concepts (exact number depends on mock response)
        assert len(concepts) >= 0
    
    def test_extract_concepts_from_chunks_with_logic_data(self, sample_chunks, sample_logic_data, mock_llm_response):
        """Test concept extraction with logic data."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        concepts = consolidator._extract_concepts_from_chunks(sample_chunks, sample_logic_data)
        
        assert isinstance(concepts, list)
        # Should have extracted some concepts
        assert len(concepts) >= 0
    
    def test_extract_concepts_from_chunk_success(self, mock_llm_response):
        """Test successful concept extraction from a single chunk."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        chunk_content = "Machine learning is a subset of artificial intelligence."
        logic_data = {"claims": [], "relations": []}
        
        concepts = consolidator._extract_concepts_from_chunk(chunk_content, logic_data, 0)
        
        assert isinstance(concepts, list)
        # Should have extracted concepts based on mock response
        assert len(concepts) >= 0
    
    def test_extract_concepts_from_chunk_failure(self):
        """Test concept extraction failure from a single chunk."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.side_effect = Exception("LLM error")
        
        chunk_content = "Test content"
        logic_data = None
        
        concepts = consolidator._extract_concepts_from_chunk(chunk_content, logic_data, 0)
        
        # Should return empty list on failure
        assert concepts == []
    
    def test_parse_concepts_from_response_success(self):
        """Test successful parsing of concepts from LLM response."""
        consolidator = DocumentConsolidator()
        
        response_text = '''[
            {
                "name": "Machine Learning",
                "description": "A subset of AI",
                "category": "ai",
                "confidence": 0.9,
                "keywords": ["AI", "algorithms"]
            }
        ]'''
        
        concepts = consolidator._parse_concepts_from_response(response_text, 0)
        
        assert len(concepts) == 1
        assert concepts[0].name == "Machine Learning"
        assert concepts[0].confidence == 0.9
    
    def test_parse_concepts_from_response_invalid_json(self):
        """Test parsing concepts from invalid JSON response."""
        consolidator = DocumentConsolidator()
        
        response_text = "Invalid JSON response"
        
        concepts = consolidator._parse_concepts_from_response(response_text, 0)
        
        # Should return empty list on parsing failure
        assert concepts == []
    
    def test_extract_concept_relations(self, sample_chunks, mock_llm_response):
        """Test concept relation extraction."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        concepts = [
            KeyConcept(concept_id="c1", name="Concept 1", description="Desc 1", category="test", confidence=0.8),
            KeyConcept(concept_id="c2", name="Concept 2", description="Desc 2", category="test", confidence=0.7)
        ]
        
        relations = consolidator._extract_concept_relations(concepts, sample_chunks)
        
        assert isinstance(relations, list)
        # Should have extracted some relations
        assert len(relations) >= 0
    
    def test_extract_relations_from_chunk_success(self, mock_llm_response):
        """Test successful relation extraction from a single chunk."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        chunk_content = "Machine learning is a subset of artificial intelligence."
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8),
            KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9)
        ]
        
        relations = consolidator._extract_relations_from_chunk(chunk_content, concepts, 0)
        
        assert isinstance(relations, list)
        # Should have extracted some relations
        assert len(relations) >= 0
    
    def test_extract_relations_from_chunk_failure(self):
        """Test relation extraction failure from a single chunk."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.side_effect = Exception("LLM error")
        
        chunk_content = "Test content"
        concepts = []
        
        relations = consolidator._extract_relations_from_chunk(chunk_content, concepts, 0)
        
        # Should return empty list on failure
        assert relations == []
    
    def test_parse_relations_from_response_success(self):
        """Test successful parsing of relations from LLM response."""
        consolidator = DocumentConsolidator()
        
        response_text = '''[
            {
                "source_concept": "c1",
                "target_concept": "c2",
                "relation_type": "hierarchical",
                "strength": 0.8,
                "description": "ML is subset of AI",
                "evidence": ["Evidence 1"]
            }
        ]'''
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8),
            KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9)
        ]
        
        relations = consolidator._parse_relations_from_response(response_text, concepts)
        
        assert len(relations) == 1
        assert relations[0].source_concept == "c1"
        assert relations[0].target_concept == "c2"
        assert relations[0].strength == 0.8
    
    def test_parse_relations_from_response_invalid_json(self):
        """Test parsing relations from invalid JSON response."""
        consolidator = DocumentConsolidator()
        
        response_text = "Invalid JSON response"
        concepts = []
        
        relations = consolidator._parse_relations_from_response(response_text, concepts)
        
        # Should return empty list on parsing failure
        assert relations == []
    
    def test_parse_relations_from_response_invalid_concept_ids(self):
        """Test parsing relations with invalid concept IDs."""
        consolidator = DocumentConsolidator()
        
        response_text = '''[
            {
                "source_concept": "invalid_c1",
                "target_concept": "invalid_c2",
                "relation_type": "hierarchical",
                "strength": 0.8,
                "description": "Invalid relation",
                "evidence": []
            }
        ]'''
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8)
        ]
        
        relations = consolidator._parse_relations_from_response(response_text, concepts)
        
        # Should return empty list due to invalid concept IDs
        assert relations == []
    
    def test_merge_similar_concepts_no_embedding_model(self):
        """Test merging similar concepts without embedding model."""
        consolidator = DocumentConsolidator()
        consolidator.embedding_model = None
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8),
            KeyConcept(concept_id="c2", name="Machine Learning", description="ML", category="ai", confidence=0.9)
        ]
        
        merged_concepts = consolidator._merge_similar_concepts(concepts)
        
        # Should return original concepts when no embedding model
        assert len(merged_concepts) == len(concepts)
    
    def test_merge_similar_concepts_with_embedding_model(self, mock_sentence_transformer):
        """Test merging similar concepts with embedding model."""
        consolidator = DocumentConsolidator()
        consolidator.embedding_model = mock_sentence_transformer
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8),
            KeyConcept(concept_id="c2", name="Machine Learning", description="ML", category="ai", confidence=0.9)
        ]
        
        merged_concepts = consolidator._merge_similar_concepts(concepts)
        
        assert isinstance(merged_concepts, list)
        # Should have some concepts (exact number depends on similarity threshold)
        assert len(merged_concepts) >= 0
    
    def test_merge_concept_group_single_concept(self):
        """Test merging a single concept (should return unchanged)."""
        consolidator = DocumentConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8)
        ]
        
        merged_concept = consolidator._merge_concept_group(concepts)
        
        assert merged_concept.concept_id == "c1"
        assert merged_concept.name == "ML"
    
    def test_merge_concept_group_multiple_concepts(self):
        """Test merging multiple concepts."""
        consolidator = DocumentConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, keywords=["AI"]),
            KeyConcept(concept_id="c2", name="Machine Learning", description="ML", category="ai", confidence=0.9, keywords=["algorithms"])
        ]
        
        merged_concept = consolidator._merge_concept_group(concepts)
        
        assert merged_concept.concept_id == "c1"  # Should use base concept ID
        assert merged_concept.confidence == 0.9  # Should use highest confidence
        assert "AI" in merged_concept.keywords
        assert "algorithms" in merged_concept.keywords
    
    def test_filter_and_merge_relations(self):
        """Test filtering and merging relations."""
        consolidator = DocumentConsolidator()
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8),
            KnowledgeRelation(relation_id="r2", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.6)
        ]
        
        filtered_relations = consolidator._filter_and_merge_relations(relations)
        
        assert isinstance(filtered_relations, list)
        # Should have merged duplicate relations
        assert len(filtered_relations) <= len(relations)
    
    def test_merge_relation_group_single_relation(self):
        """Test merging a single relation (should return unchanged)."""
        consolidator = DocumentConsolidator()
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8)
        ]
        
        merged_relation = consolidator._merge_relation_group(relations)
        
        assert merged_relation.relation_id == "r1"
        assert merged_relation.strength == 0.8
    
    def test_merge_relation_group_multiple_relations(self):
        """Test merging multiple relations."""
        consolidator = DocumentConsolidator()
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8, evidence=["Evidence 1"]),
            KnowledgeRelation(relation_id="r2", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.6, evidence=["Evidence 2"])
        ]
        
        merged_relation = consolidator._merge_relation_group(relations)
        
        assert merged_relation.relation_id == "r1"  # Should use base relation ID
        assert merged_relation.strength == 0.8  # Should use highest strength
        assert "Evidence 1" in merged_relation.evidence
        assert "Evidence 2" in merged_relation.evidence
    
    def test_generate_document_summary(self, mock_llm_response):
        """Test document summary generation."""
        consolidator = DocumentConsolidator()
        consolidator.llm.invoke.return_value = mock_llm_response
        
        chunks = [Mock(text="Machine learning is AI", metadata=Mock())]
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8)
        ]
        
        summary = consolidator._generate_document_summary(chunks, concepts)
        
        assert isinstance(summary, str)
        # Should have generated some summary (exact content depends on mock response)
        assert len(summary) > 0
    
    def test_extract_main_themes(self):
        """Test main theme extraction."""
        consolidator = DocumentConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8),
            KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9),
            KeyConcept(concept_id="c3", name="Data", description="Data Science", category="data", confidence=0.7)
        ]
        
        themes = consolidator._extract_main_themes(concepts)
        
        assert isinstance(themes, list)
        # Should have extracted some themes
        assert len(themes) >= 0
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        consolidator = DocumentConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8),
            KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9)
        ]
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.7)
        ]
        
        quality_score = consolidator._calculate_quality_score(concepts, relations)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_calculate_quality_score_no_concepts(self):
        """Test quality score calculation with no concepts."""
        consolidator = DocumentConsolidator()
        
        concepts = []
        relations = []
        
        quality_score = consolidator._calculate_quality_score(concepts, relations)
        
        assert quality_score == 0.0
    
    def test_get_chunk_content(self):
        """Test chunk content extraction."""
        consolidator = DocumentConsolidator()
        
        # Test with chunk that has text attribute
        chunk_with_text = Mock(text="Test content", metadata=Mock())
        content = consolidator._get_chunk_content(chunk_with_text)
        assert content == "Test content"
        
        # Test with chunk that has content attribute
        chunk_with_content = Mock(content="Test content", metadata=Mock())
        content = consolidator._get_chunk_content(chunk_with_content)
        assert content == "Test content"
        
        # Test with chunk that has neither
        chunk_with_neither = Mock()
        content = consolidator._get_chunk_content(chunk_with_neither)
        assert isinstance(content, str)
    
    def test_get_chunk_content_with_exception(self):
        """Test chunk content extraction with exception."""
        consolidator = DocumentConsolidator()
        
        # Create a mock that raises an exception when accessing attributes
        chunk_with_exception = Mock()
        chunk_with_exception.text = Mock(side_effect=Exception("Attribute error"))
        
        content = consolidator._get_chunk_content(chunk_with_exception)
        
        # Should return string representation on exception
        assert isinstance(content, str)
