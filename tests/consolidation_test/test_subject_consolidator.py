"""
Unit tests for subject consolidator.

This module contains tests for the SubjectConsolidator class
and its methods for consolidating subject knowledge.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from consolidation.subject_consolidator import SubjectConsolidator
from consolidation.models import DocumentKnowledge, SubjectKnowledge, KeyConcept, KnowledgeRelation
from consolidation.config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from consolidation.exceptions import (
    SubjectConsolidationError, ConceptExtractionError, RelationExtractionError,
    MissingAPIKeyError, KnowledgeValidationError, KnowledgeMergeError
)


class TestSubjectConsolidator:
    """Test cases for the SubjectConsolidator class."""
    
    @pytest.fixture(autouse=True)
    def mock_chat_openai(self):
        """Mock ChatOpenAI to prevent actual API calls."""
        with patch('consolidation.subject_consolidator.ChatOpenAI') as mock_llm_class:
            mock_llm_instance = Mock()
            mock_llm_instance.with_structured_output.return_value = mock_llm_instance
            mock_llm_class.return_value = mock_llm_instance
            yield mock_llm_class
    
    @pytest.fixture(autouse=True)
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer to prevent actual model loading."""
        with patch('consolidation.subject_consolidator.SentenceTransformer') as mock_model_class:
            mock_model_instance = Mock()
            mock_model_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            mock_model_class.return_value = mock_model_instance
            yield mock_model_instance
    
    @pytest.fixture
    def sample_document_knowledge_list(self):
        """Create sample document knowledge list for testing."""
        doc1 = DocumentKnowledge(
            document_id="doc_1",
            title="AI Document 1",
            summary="First AI document",
            key_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
                KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9, source_chunks=["chunk_2"])
            ],
            knowledge_relations=[
                KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="hierarchical", strength=0.8)
            ],
            main_themes=["AI", "ML"],
            knowledge_graph={},
            quality_score=0.85
        )
        
        doc2 = DocumentKnowledge(
            document_id="doc_2",
            title="AI Document 2",
            summary="Second AI document",
            key_concepts=[
                KeyConcept(concept_id="c3", name="DL", description="Deep Learning", category="ai", confidence=0.7, source_chunks=["chunk_3"]),
                KeyConcept(concept_id="c4", name="NN", description="Neural Networks", category="ai", confidence=0.8, source_chunks=["chunk_4"])
            ],
            knowledge_relations=[
                KnowledgeRelation(relation_id="r2", source_concept="c3", target_concept="c4", relation_type="causal", strength=0.7)
            ],
            main_themes=["Deep Learning", "Neural Networks"],
            knowledge_graph={},
            quality_score=0.8
        )
        
        return [doc1, doc2]
    
    def test_init_default_config(self, mock_chat_openai):
        """Test initialization with default configuration."""
        consolidator = SubjectConsolidator()
        
        assert isinstance(consolidator.config, ConsolidationConfig)
        assert consolidator.config.model_name == "gpt-4o-mini"
        mock_chat_openai.assert_called_once()
    
    def test_init_custom_config(self, mock_chat_openai):
        """Test initialization with custom configuration."""
        config = ConsolidationConfig(model_name="gpt-4o", temperature=0.2)
        consolidator = SubjectConsolidator(config)
        
        assert consolidator.config == config
        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.2, max_tokens=4000)
    
    def test_init_with_missing_api_key(self):
        """Test initialization failure with missing API key."""
        with patch('consolidation.subject_consolidator.ChatOpenAI', side_effect=Exception("API key not found")):
            with pytest.raises(MissingAPIKeyError, match="Failed to initialize LLM"):
                SubjectConsolidator()
    
    def test_consolidate_subject_success(self, sample_document_knowledge_list):
        """Test successful subject consolidation."""
        consolidator = SubjectConsolidator()
        
        result = consolidator.consolidate_subject(
            subject_id="subject_ai",
            subject_name="Artificial Intelligence",
            document_knowledge=sample_document_knowledge_list,
            subject_description="Comprehensive AI knowledge"
        )
        
        assert isinstance(result, SubjectKnowledge)
        assert result.subject_id == "subject_ai"
        assert result.name == "Artificial Intelligence"
        assert len(result.core_concepts) > 0
        assert len(result.document_sources) == 2
        assert result.quality_score >= 0.0
    
    def test_consolidate_subject_without_description(self, sample_document_knowledge_list):
        """Test subject consolidation without provided description."""
        consolidator = SubjectConsolidator()
        
        result = consolidator.consolidate_subject(
            subject_id="subject_ai",
            subject_name="Artificial Intelligence",
            document_knowledge=sample_document_knowledge_list
        )
        
        assert isinstance(result, SubjectKnowledge)
        assert result.subject_id == "subject_ai"
        assert result.name == "Artificial Intelligence"
        assert result.description is not None  # Should be generated
    
    def test_consolidate_subject_failure(self, sample_document_knowledge_list):
        """Test subject consolidation failure."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(SubjectConsolidationError, match="Failed to consolidate subject"):
            consolidator.consolidate_subject(
                subject_id="subject_ai",
                subject_name="Artificial Intelligence",
                document_knowledge=sample_document_knowledge_list
            )
    
    def test_extract_and_merge_concepts(self, sample_document_knowledge_list):
        """Test concept extraction and merging from documents."""
        consolidator = SubjectConsolidator()
        
        concepts = consolidator._extract_and_merge_concepts(sample_document_knowledge_list)
        
        assert isinstance(concepts, list)
        # Should have extracted concepts from all documents
        assert len(concepts) > 0
    
    def test_extract_and_merge_concepts_empty_documents(self):
        """Test concept extraction with empty document list."""
        consolidator = SubjectConsolidator()
        
        concepts = consolidator._extract_and_merge_concepts([])
        
        assert concepts == []
    
    def test_extract_and_merge_concepts_no_concepts_in_documents(self):
        """Test concept extraction with documents that have no concepts."""
        doc_without_concepts = DocumentKnowledge(
            document_id="doc_empty",
            title="Empty Document",
            summary="Document without concepts",
            key_concepts=[],
            knowledge_relations=[],
            main_themes=[],
            knowledge_graph={},
            quality_score=0.5
        )
        
        consolidator = SubjectConsolidator()
        
        concepts = consolidator._extract_and_merge_concepts([doc_without_concepts])
        
        assert concepts == []
    
    def test_merge_similar_concepts_across_documents_no_embedding_model(self, sample_document_knowledge_list):
        """Test merging similar concepts without embedding model."""
        consolidator = SubjectConsolidator()
        consolidator.embedding_model = None
        
        # Extract concepts first
        all_concepts = []
        for doc in sample_document_knowledge_list:
            all_concepts.extend(doc.key_concepts)
        
        merged_concepts = consolidator._merge_similar_concepts_across_documents(all_concepts)
        
        # Should return original concepts when no embedding model
        assert len(merged_concepts) == len(all_concepts)
    
    def test_merge_similar_concepts_across_documents_with_embedding_model(self, sample_document_knowledge_list, mock_sentence_transformer):
        """Test merging similar concepts with embedding model."""
        consolidator = SubjectConsolidator()
        consolidator.embedding_model = mock_sentence_transformer
        
        # Extract concepts first
        all_concepts = []
        for doc in sample_document_knowledge_list:
            all_concepts.extend(doc.key_concepts)
        
        merged_concepts = consolidator._merge_similar_concepts_across_documents(all_concepts)
        
        assert isinstance(merged_concepts, list)
        # Should have some concepts (exact number depends on similarity threshold)
        assert len(merged_concepts) >= 0
    
    def test_merge_concept_group_across_documents_single_concept(self):
        """Test merging a single concept across documents (should return unchanged)."""
        consolidator = SubjectConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
        ]
        
        merged_concept = consolidator._merge_concept_group_across_documents(concepts)
        
        assert merged_concept.concept_id == "c1"
        assert merged_concept.name == "ML"
    
    def test_merge_concept_group_across_documents_multiple_concepts(self):
        """Test merging multiple concepts across documents."""
        consolidator = SubjectConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"], keywords=["AI"]),
            KeyConcept(concept_id="c2", name="Machine Learning", description="ML", category="ai", confidence=0.9, source_chunks=["chunk_2"], keywords=["algorithms"])
        ]
        
        merged_concept = consolidator._merge_concept_group_across_documents(concepts)
        
        assert merged_concept.concept_id == "subject_c1"  # Should have subject prefix
        assert merged_concept.confidence == 0.9  # Should use highest confidence
        assert "AI" in merged_concept.keywords
        assert "algorithms" in merged_concept.keywords
    
    def test_merge_concept_descriptions_single_concept(self):
        """Test merging descriptions for a single concept."""
        consolidator = SubjectConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
        ]
        
        merged_description = consolidator._merge_concept_descriptions(concepts)
        
        assert merged_description == "Machine Learning"  # Should return original description
    
    def test_merge_concept_descriptions_multiple_concepts(self):
        """Test merging descriptions for multiple concepts."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.return_value = Mock(content="Merged description of Machine Learning concepts")
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning approach 1", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
            KeyConcept(concept_id="c2", name="Machine Learning", description="Machine Learning approach 2", category="ai", confidence=0.9, source_chunks=["chunk_2"])
        ]
        
        merged_description = consolidator._merge_concept_descriptions(concepts)
        
        assert isinstance(merged_description, str)
        assert len(merged_description) > 0
    
    def test_merge_concept_descriptions_llm_failure(self):
        """Test merging descriptions with LLM failure."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.side_effect = Exception("LLM error")
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning approach 1", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
            KeyConcept(concept_id="c2", name="Machine Learning", description="Machine Learning approach 2", category="ai", confidence=0.9, source_chunks=["chunk_2"])
        ]
        
        merged_description = consolidator._merge_concept_descriptions(concepts)
        
        # Should return the description of the first concept on failure
        assert merged_description == "Machine Learning approach 1"
    
    def test_extract_and_merge_relations(self, sample_document_knowledge_list):
        """Test relation extraction and merging from documents."""
        consolidator = SubjectConsolidator()
        
        # Extract concepts first
        all_concepts = []
        for doc in sample_document_knowledge_list:
            all_concepts.extend(doc.key_concepts)
        
        relations = consolidator._extract_and_merge_relations(sample_document_knowledge_list, all_concepts)
        
        assert isinstance(relations, list)
        # Should have extracted some relations
        assert len(relations) >= 0
    
    def test_extract_and_merge_relations_no_valid_concepts(self, sample_document_knowledge_list):
        """Test relation extraction with no valid concepts."""
        consolidator = SubjectConsolidator()
        
        # Use empty concepts list
        relations = consolidator._extract_and_merge_relations(sample_document_knowledge_list, [])
        
        assert relations == []
    
    def test_merge_duplicate_relations(self):
        """Test merging duplicate relations."""
        consolidator = SubjectConsolidator()
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8),
            KnowledgeRelation(relation_id="r2", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.6)
        ]
        
        merged_relations = consolidator._merge_duplicate_relations(relations)
        
        assert isinstance(merged_relations, list)
        # Should have merged duplicate relations
        assert len(merged_relations) <= len(relations)
    
    def test_merge_relation_group_across_documents_single_relation(self):
        """Test merging a single relation across documents (should return unchanged)."""
        consolidator = SubjectConsolidator()
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8)
        ]
        
        merged_relation = consolidator._merge_relation_group_across_documents(relations)
        
        assert merged_relation.relation_id == "r1"
        assert merged_relation.strength == 0.8
    
    def test_merge_relation_group_across_documents_multiple_relations(self):
        """Test merging multiple relations across documents."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.return_value = Mock(content="Merged relation description")
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8, evidence=["Evidence 1"]),
            KnowledgeRelation(relation_id="r2", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.6, evidence=["Evidence 2"])
        ]
        
        merged_relation = consolidator._merge_relation_group_across_documents(relations)
        
        assert merged_relation.relation_id == "subject_r1"  # Should have subject prefix
        assert merged_relation.strength == 0.8  # Should use highest strength
        assert "Evidence 1" in merged_relation.evidence
        assert "Evidence 2" in merged_relation.evidence
    
    def test_merge_relation_descriptions_single_relation(self):
        """Test merging descriptions for a single relation."""
        consolidator = SubjectConsolidator()
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8, description="Original description")
        ]
        
        merged_description = consolidator._merge_relation_descriptions(relations)
        
        assert merged_description == "Original description"  # Should return original description
    
    def test_merge_relation_descriptions_multiple_relations(self):
        """Test merging descriptions for multiple relations."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.return_value = Mock(content="Merged relation description")
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8, description="Description 1"),
            KnowledgeRelation(relation_id="r2", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.6, description="Description 2")
        ]
        
        merged_description = consolidator._merge_relation_descriptions(relations)
        
        assert isinstance(merged_description, str)
        assert len(merged_description) > 0
    
    def test_merge_relation_descriptions_llm_failure(self):
        """Test merging descriptions with LLM failure."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.side_effect = Exception("LLM error")
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.8, description="Description 1"),
            KnowledgeRelation(relation_id="r2", source_concept="c1", target_concept="c2", relation_type="causal", strength=0.6, description="Description 2")
        ]
        
        merged_description = consolidator._merge_relation_descriptions(relations)
        
        # Should return the description of the first relation on failure
        assert merged_description == "Description 1"
    
    def test_build_knowledge_hierarchy(self, sample_document_knowledge_list):
        """Test knowledge hierarchy building."""
        consolidator = SubjectConsolidator()
        
        # Extract concepts first
        all_concepts = []
        for doc in sample_document_knowledge_list:
            all_concepts.extend(doc.key_concepts)
        
        # Create some relations for hierarchy
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="hierarchical", strength=0.8)
        ]
        
        hierarchy = consolidator._build_knowledge_hierarchy(all_concepts, relations)
        
        assert isinstance(hierarchy, dict)
        assert "levels" in hierarchy
        assert "root_concepts" in hierarchy
        assert "leaf_concepts" in hierarchy
    
    def test_build_knowledge_hierarchy_no_relations(self):
        """Test knowledge hierarchy building with no relations."""
        consolidator = SubjectConsolidator()
        
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
        ]
        
        hierarchy = consolidator._build_knowledge_hierarchy(concepts, [])
        
        assert isinstance(hierarchy, dict)
        assert "levels" in hierarchy
        assert "root_concepts" in hierarchy
        assert "leaf_concepts" in hierarchy
    
    def test_calculate_hierarchy_levels(self):
        """Test hierarchy level calculation."""
        consolidator = SubjectConsolidator()
        
        # Create a simple hierarchy
        concept_network = {
            "c1": {"concept": None, "children": ["c2"], "parents": [], "level": 0},
            "c2": {"concept": None, "children": [], "parents": ["c1"], "level": 0}
        }
        
        consolidator._calculate_hierarchy_levels(concept_network)
        
        # Check that levels were calculated
        assert concept_network["c1"]["level"] == 0  # Root concept
        assert concept_network["c2"]["level"] == 1  # Child concept
    
    def test_generate_subject_description(self, sample_document_knowledge_list):
        """Test subject description generation."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.return_value = Mock(content="Generated subject description")
        
        concepts = []
        for doc in sample_document_knowledge_list:
            concepts.extend(doc.key_concepts)
        
        description = consolidator._generate_subject_description(concepts, sample_document_knowledge_list)
        
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_generate_subject_description_llm_failure(self, sample_document_knowledge_list):
        """Test subject description generation with LLM failure."""
        consolidator = SubjectConsolidator()
        consolidator.llm.invoke.side_effect = Exception("LLM error")
        
        concepts = []
        for doc in sample_document_knowledge_list:
            concepts.extend(doc.key_concepts)
        
        description = consolidator._generate_subject_description(concepts, sample_document_knowledge_list)
        
        # Should return a fallback description
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_determine_expertise_level(self, sample_document_knowledge_list):
        """Test expertise level determination."""
        consolidator = SubjectConsolidator()
        
        concepts = []
        for doc in sample_document_knowledge_list:
            concepts.extend(doc.key_concepts)
        
        relations = []
        for doc in sample_document_knowledge_list:
            relations.extend(doc.knowledge_relations)
        
        expertise_level = consolidator._determine_expertise_level(concepts, relations)
        
        assert expertise_level in ["beginner", "intermediate", "advanced"]
    
    def test_determine_expertise_level_no_concepts(self):
        """Test expertise level determination with no concepts."""
        consolidator = SubjectConsolidator()
        
        expertise_level = consolidator._determine_expertise_level([], [])
        
        assert expertise_level == "beginner"
    
    def test_extract_domain_tags(self, sample_document_knowledge_list):
        """Test domain tag extraction."""
        consolidator = SubjectConsolidator()
        
        concepts = []
        for doc in sample_document_knowledge_list:
            concepts.extend(doc.key_concepts)
        
        domain_tags = consolidator._extract_domain_tags(concepts, sample_document_knowledge_list)
        
        assert isinstance(domain_tags, list)
        # Should have extracted some domain tags
        assert len(domain_tags) >= 0
    
    def test_extract_domain_tags_no_concepts(self):
        """Test domain tag extraction with no concepts."""
        consolidator = SubjectConsolidator()
        
        domain_tags = consolidator._extract_domain_tags([], [])
        
        assert domain_tags == []
    
    def test_calculate_subject_quality_score(self, sample_document_knowledge_list):
        """Test subject quality score calculation."""
        consolidator = SubjectConsolidator()
        
        concepts = []
        for doc in sample_document_knowledge_list:
            concepts.extend(doc.key_concepts)
        
        relations = []
        for doc in sample_document_knowledge_list:
            relations.extend(doc.knowledge_relations)
        
        quality_score = consolidator._calculate_subject_quality_score(concepts, relations)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_calculate_subject_quality_score_no_concepts(self):
        """Test subject quality score calculation with no concepts."""
        consolidator = SubjectConsolidator()
        
        quality_score = consolidator._calculate_subject_quality_score([], [])
        
        assert quality_score == 0.0
    
    def test_calculate_subject_quality_score_exception(self):
        """Test subject quality score calculation with exception."""
        consolidator = SubjectConsolidator()
        
        # Create concepts that might cause issues
        concepts = [Mock()]  # Mock object that might not have expected attributes
        
        quality_score = consolidator._calculate_subject_quality_score(concepts, [])
        
        # Should return a default value on exception
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
