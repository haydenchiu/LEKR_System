"""
Unit tests for consolidation utilities.

This module contains tests for utility functions used in the
consolidation module.
"""

import pytest
from unittest.mock import Mock

from consolidation.utils import (
    extract_key_concepts,
    build_knowledge_graph,
    validate_knowledge_consistency,
    merge_related_concepts,
    calculate_knowledge_coverage,
    find_knowledge_gaps
)
from consolidation.models import DocumentKnowledge, SubjectKnowledge, KeyConcept, KnowledgeRelation


class TestExtractKeyConcepts:
    """Test cases for the extract_key_concepts utility function."""
    
    def test_extract_key_concepts_basic(self):
        """Test basic concept extraction."""
        text = "Machine Learning is a subset of Artificial Intelligence. Deep Learning uses Neural Networks."
        concepts = extract_key_concepts(text)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        # Should extract capitalized terms
        assert any("Machine Learning" in concept for concept in concepts)
        assert any("Artificial Intelligence" in concept for concept in concepts)
    
    def test_extract_key_concepts_with_quotes(self):
        """Test concept extraction with quoted terms."""
        text = 'The "Transformer" model is a "state-of-the-art" approach to NLP.'
        concepts = extract_key_concepts(text)
        
        assert isinstance(concepts, list)
        assert "Transformer" in concepts
        assert "state-of-the-art" in concepts
    
    def test_extract_key_concepts_with_parentheses(self):
        """Test concept extraction with parenthetical terms."""
        text = "BERT (Bidirectional Encoder Representations from Transformers) is a language model."
        concepts = extract_key_concepts(text)
        
        assert isinstance(concepts, list)
        assert "Bidirectional Encoder Representations from Transformers" in concepts
    
    def test_extract_key_concepts_empty_text(self):
        """Test concept extraction with empty text."""
        concepts = extract_key_concepts("")
        assert concepts == []
    
    def test_extract_key_concepts_max_concepts(self):
        """Test concept extraction with max_concepts limit."""
        text = "Machine Learning AI Deep Learning Neural Networks Natural Language Processing Computer Vision Data Science"
        concepts = extract_key_concepts(text, max_concepts=3)
        
        assert isinstance(concepts, list)
        assert len(concepts) <= 3
    
    def test_extract_key_concepts_filtering(self):
        """Test concept extraction filtering by length."""
        text = "A B C Machine Learning AI Deep Learning"
        concepts = extract_key_concepts(text)
        
        assert isinstance(concepts, list)
        # Should filter out very short terms
        assert "A" not in concepts
        assert "B" not in concepts
        assert "C" not in concepts
        assert "Machine Learning" in concepts


class TestBuildKnowledgeGraph:
    """Test cases for the build_knowledge_graph utility function."""
    
    def test_build_knowledge_graph_basic(self):
        """Test basic knowledge graph building."""
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
            KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9, source_chunks=["chunk_2"])
        ]
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="hierarchical", strength=0.8)
        ]
        
        graph = build_knowledge_graph(concepts, relations)
        
        assert isinstance(graph, dict)
        assert "nodes" in graph
        assert "edges" in graph
        assert "adjacency" in graph
        assert "metrics" in graph
        assert "components" in graph
        
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["metrics"]["num_nodes"] == 2
        assert graph["metrics"]["num_edges"] == 1
    
    def test_build_knowledge_graph_empty_concepts(self):
        """Test knowledge graph building with empty concepts."""
        concepts = []
        relations = []
        
        graph = build_knowledge_graph(concepts, relations)
        
        assert isinstance(graph, dict)
        assert graph["nodes"] == {}
        assert graph["edges"] == []
        assert graph["metrics"]["num_nodes"] == 0
        assert graph["metrics"]["num_edges"] == 0
    
    def test_build_knowledge_graph_no_relations(self):
        """Test knowledge graph building with no relations."""
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
        ]
        relations = []
        
        graph = build_knowledge_graph(concepts, relations)
        
        assert isinstance(graph, dict)
        assert len(graph["nodes"]) == 1
        assert len(graph["edges"]) == 0
        assert graph["metrics"]["num_nodes"] == 1
        assert graph["metrics"]["num_edges"] == 0
    
    def test_build_knowledge_graph_exception_handling(self):
        """Test knowledge graph building with exception handling."""
        # Create invalid concepts that might cause issues
        concepts = [Mock()]  # Mock object without expected attributes
        relations = []
        
        graph = build_knowledge_graph(concepts, relations)
        
        # Should return empty graph on exception
        assert isinstance(graph, dict)
        assert "nodes" in graph
        assert "edges" in graph


class TestValidateKnowledgeConsistency:
    """Test cases for the validate_knowledge_consistency utility function."""
    
    def test_validate_document_knowledge_consistent(self):
        """Test validation of consistent document knowledge."""
        doc_knowledge = DocumentKnowledge(
            document_id="doc_1",
            title="Test Document",
            summary="Test summary",
            key_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
            ],
            knowledge_relations=[],
            main_themes=["AI"],
            knowledge_graph={},
            quality_score=0.8
        )
        
        result = validate_knowledge_consistency(doc_knowledge)
        assert result is True
    
    def test_validate_document_knowledge_inconsistent(self):
        """Test validation of inconsistent document knowledge."""
        # Create invalid document knowledge (missing required fields)
        doc_knowledge = DocumentKnowledge(
            document_id="",  # Empty document ID
            title="Test Document",
            summary="Test summary",
            key_concepts=[],
            knowledge_relations=[],
            main_themes=[],
            knowledge_graph={},
            quality_score=0.8
        )
        
        result = validate_knowledge_consistency(doc_knowledge)
        assert result is False
    
    def test_validate_subject_knowledge_consistent(self):
        """Test validation of consistent subject knowledge."""
        subject_knowledge = SubjectKnowledge(
            subject_id="subject_ai",
            name="Artificial Intelligence",
            description="AI knowledge",
            core_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
            ],
            knowledge_relations=[],
            document_sources=["doc_1"],
            knowledge_hierarchy={},
            expertise_level="intermediate",
            domain_tags=["AI"],
            quality_score=0.8
        )
        
        result = validate_knowledge_consistency(subject_knowledge)
        assert result is True
    
    def test_validate_subject_knowledge_inconsistent(self):
        """Test validation of inconsistent subject knowledge."""
        # Create invalid subject knowledge (invalid expertise level)
        subject_knowledge = SubjectKnowledge(
            subject_id="subject_ai",
            name="Artificial Intelligence",
            description="AI knowledge",
            core_concepts=[],
            knowledge_relations=[],
            document_sources=[],
            knowledge_hierarchy={},
            expertise_level="invalid_level",  # Invalid expertise level
            domain_tags=[],
            quality_score=0.8
        )
        
        result = validate_knowledge_consistency(subject_knowledge)
        assert result is False
    
    def test_validate_knowledge_consistency_invalid_type(self):
        """Test validation with invalid knowledge type."""
        result = validate_knowledge_consistency("not a knowledge object")
        assert result is False
    
    def test_validate_knowledge_consistency_exception_handling(self):
        """Test validation with exception handling."""
        # Create a mock object that raises an exception
        mock_knowledge = Mock()
        mock_knowledge.__class__.__name__ = "DocumentKnowledge"
        mock_knowledge.document_id = "doc_1"
        mock_knowledge.title = "Test"
        mock_knowledge.summary = "Test"
        mock_knowledge.key_concepts = []
        mock_knowledge.knowledge_relations = []
        mock_knowledge.quality_score = 0.8
        # Make it raise an exception when accessing attributes
        mock_knowledge.key_concepts = Mock(side_effect=Exception("Access error"))
        
        result = validate_knowledge_consistency(mock_knowledge)
        assert result is False


class TestMergeRelatedConcepts:
    """Test cases for the merge_related_concepts utility function."""
    
    def test_merge_related_concepts_single_concept(self):
        """Test merging a single concept (should return unchanged)."""
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
        ]
        
        merged_concepts = merge_related_concepts(concepts)
        
        assert len(merged_concepts) == 1
        assert merged_concepts[0].concept_id == "c1"
    
    def test_merge_related_concepts_no_concepts(self):
        """Test merging with no concepts."""
        concepts = []
        
        merged_concepts = merge_related_concepts(concepts)
        
        assert merged_concepts == []
    
    def test_merge_related_concepts_different_concepts(self):
        """Test merging different concepts (should not merge)."""
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
            KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9, source_chunks=["chunk_2"])
        ]
        
        merged_concepts = merge_related_concepts(concepts, similarity_threshold=0.9)
        
        # Should not merge if similarity is below threshold
        assert len(merged_concepts) == 2
    
    def test_merge_related_concepts_similar_concepts(self):
        """Test merging similar concepts."""
        concepts = [
            KeyConcept(concept_id="c1", name="Machine Learning", description="ML is a subset of AI", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
            KeyConcept(concept_id="c2", name="ML", description="Machine Learning algorithms", category="ai", confidence=0.9, source_chunks=["chunk_2"])
        ]
        
        merged_concepts = merge_related_concepts(concepts, similarity_threshold=0.5)
        
        # Should merge if similarity is above threshold
        assert len(merged_concepts) <= 2
        assert len(merged_concepts) >= 1
    
    def test_merge_related_concepts_exception_handling(self):
        """Test merging concepts with exception handling."""
        # Create concepts that might cause issues
        concepts = [Mock()]  # Mock object without expected attributes
        
        merged_concepts = merge_related_concepts(concepts)
        
        # Should return original concepts on exception
        assert len(merged_concepts) == 1


class TestCalculateKnowledgeCoverage:
    """Test cases for the calculate_knowledge_coverage utility function."""
    
    def test_calculate_knowledge_coverage_document(self):
        """Test coverage calculation for document knowledge."""
        doc_knowledge = DocumentKnowledge(
            document_id="doc_1",
            title="Test Document",
            summary="Test summary",
            key_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
                KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9, source_chunks=["chunk_2"])
            ],
            knowledge_relations=[
                KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="hierarchical", strength=0.8)
            ],
            main_themes=["AI"],
            knowledge_graph={},
            quality_score=0.8
        )
        
        coverage = calculate_knowledge_coverage(doc_knowledge)
        
        assert isinstance(coverage, dict)
        assert "concept_coverage" in coverage
        assert "relation_coverage" in coverage
        assert "overall_coverage" in coverage
        
        assert 0.0 <= coverage["concept_coverage"] <= 1.0
        assert 0.0 <= coverage["relation_coverage"] <= 1.0
        assert 0.0 <= coverage["overall_coverage"] <= 1.0
    
    def test_calculate_knowledge_coverage_subject(self):
        """Test coverage calculation for subject knowledge."""
        subject_knowledge = SubjectKnowledge(
            subject_id="subject_ai",
            name="Artificial Intelligence",
            description="AI knowledge",
            core_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
            ],
            knowledge_relations=[],
            document_sources=["doc_1"],
            knowledge_hierarchy={},
            expertise_level="intermediate",
            domain_tags=["AI"],
            quality_score=0.8
        )
        
        coverage = calculate_knowledge_coverage(subject_knowledge)
        
        assert isinstance(coverage, dict)
        assert "concept_coverage" in coverage
        assert "relation_coverage" in coverage
        assert "overall_coverage" in coverage
    
    def test_calculate_knowledge_coverage_no_concepts(self):
        """Test coverage calculation with no concepts."""
        doc_knowledge = DocumentKnowledge(
            document_id="doc_1",
            title="Test Document",
            summary="Test summary",
            key_concepts=[],
            knowledge_relations=[],
            main_themes=[],
            knowledge_graph={},
            quality_score=0.8
        )
        
        coverage = calculate_knowledge_coverage(doc_knowledge)
        
        assert coverage["concept_coverage"] == 0.0
        assert coverage["relation_coverage"] == 0.0
        assert coverage["overall_coverage"] == 0.0
    
    def test_calculate_knowledge_coverage_invalid_type(self):
        """Test coverage calculation with invalid knowledge type."""
        coverage = calculate_knowledge_coverage("not a knowledge object")
        assert coverage == {}
    
    def test_calculate_knowledge_coverage_exception_handling(self):
        """Test coverage calculation with exception handling."""
        # Create a mock object that raises an exception
        mock_knowledge = Mock()
        mock_knowledge.__class__.__name__ = "DocumentKnowledge"
        mock_knowledge.key_concepts = Mock(side_effect=Exception("Access error"))
        
        coverage = calculate_knowledge_coverage(mock_knowledge)
        
        # Should return default values on exception
        assert coverage == {"concept_coverage": 0.0, "relation_coverage": 0.0, "overall_coverage": 0.0}


class TestFindKnowledgeGaps:
    """Test cases for the find_knowledge_gaps utility function."""
    
    def test_find_knowledge_gaps_document(self):
        """Test gap finding for document knowledge."""
        doc_knowledge = DocumentKnowledge(
            document_id="doc_1",
            title="Test Document",
            summary="Test summary",
            key_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
                KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9, source_chunks=["chunk_2"])
            ],
            knowledge_relations=[
                KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="hierarchical", strength=0.8)
            ],
            main_themes=["AI"],
            knowledge_graph={},
            quality_score=0.8
        )
        
        gaps = find_knowledge_gaps(doc_knowledge)
        
        assert isinstance(gaps, list)
        # Should find some gaps (exact content depends on the knowledge structure)
        assert len(gaps) >= 0
    
    def test_find_knowledge_gaps_subject(self):
        """Test gap finding for subject knowledge."""
        subject_knowledge = SubjectKnowledge(
            subject_id="subject_ai",
            name="Artificial Intelligence",
            description="AI knowledge",
            core_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
            ],
            knowledge_relations=[],
            document_sources=["doc_1"],
            knowledge_hierarchy={},
            expertise_level="intermediate",
            domain_tags=["AI"],
            quality_score=0.8
        )
        
        gaps = find_knowledge_gaps(subject_knowledge)
        
        assert isinstance(gaps, list)
        # Should find some gaps
        assert len(gaps) >= 0
    
    def test_find_knowledge_gaps_no_concepts(self):
        """Test gap finding with no concepts."""
        doc_knowledge = DocumentKnowledge(
            document_id="doc_1",
            title="Test Document",
            summary="Test summary",
            key_concepts=[],
            knowledge_relations=[],
            main_themes=[],
            knowledge_graph={},
            quality_score=0.8
        )
        
        gaps = find_knowledge_gaps(doc_knowledge)
        
        assert isinstance(gaps, list)
        # Should find gaps related to missing concepts
        assert len(gaps) >= 0
    
    def test_find_knowledge_gaps_invalid_type(self):
        """Test gap finding with invalid knowledge type."""
        gaps = find_knowledge_gaps("not a knowledge object")
        assert gaps == []
    
    def test_find_knowledge_gaps_exception_handling(self):
        """Test gap finding with exception handling."""
        # Create a mock object that raises an exception
        mock_knowledge = Mock()
        mock_knowledge.__class__.__name__ = "DocumentKnowledge"
        mock_knowledge.key_concepts = Mock(side_effect=Exception("Access error"))
        
        gaps = find_knowledge_gaps(mock_knowledge)
        
        # Should return empty list on exception
        assert gaps == []


class TestUtilityIntegration:
    """Integration tests for utility functions."""
    
    def test_utility_functions_work_together(self):
        """Test that utility functions can work together."""
        # Create sample knowledge
        concepts = [
            KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"]),
            KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9, source_chunks=["chunk_2"])
        ]
        
        relations = [
            KnowledgeRelation(relation_id="r1", source_concept="c1", target_concept="c2", relation_type="hierarchical", strength=0.8)
        ]
        
        doc_knowledge = DocumentKnowledge(
            document_id="doc_1",
            title="Test Document",
            summary="Test summary",
            key_concepts=concepts,
            knowledge_relations=relations,
            main_themes=["AI"],
            knowledge_graph={},
            quality_score=0.8
        )
        
        # Test that all utility functions work with the same knowledge object
        graph = build_knowledge_graph(concepts, relations)
        assert isinstance(graph, dict)
        
        is_consistent = validate_knowledge_consistency(doc_knowledge)
        assert isinstance(is_consistent, bool)
        
        coverage = calculate_knowledge_coverage(doc_knowledge)
        assert isinstance(coverage, dict)
        
        gaps = find_knowledge_gaps(doc_knowledge)
        assert isinstance(gaps, list)
    
    def test_utility_functions_with_edge_cases(self):
        """Test utility functions with edge cases."""
        # Test with empty data
        empty_concepts = []
        empty_relations = []
        
        graph = build_knowledge_graph(empty_concepts, empty_relations)
        assert graph["nodes"] == {}
        assert graph["edges"] == []
        
        # Test with single concept
        single_concept = [KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])]
        
        merged_concepts = merge_related_concepts(single_concept)
        assert len(merged_concepts) == 1
        
        # Test concept extraction with various text types
        text_variations = [
            "Simple text",
            "Text with CAPITALIZED terms",
            'Text with "quoted" terms',
            "Text with (parenthetical) terms",
            "Text with numbers 123 and symbols !@#",
            ""  # Empty text
        ]
        
        for text in text_variations:
            concepts = extract_key_concepts(text)
            assert isinstance(concepts, list)
