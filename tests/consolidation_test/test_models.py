"""
Unit tests for consolidation models.

This module contains tests for all Pydantic models used in the
consolidation module, including KeyConcept, KnowledgeRelation,
DocumentKnowledge, and SubjectKnowledge.
"""

import pytest
from datetime import datetime
from consolidation.models import KeyConcept, KnowledgeRelation, DocumentKnowledge, SubjectKnowledge


class TestKeyConcept:
    """Test cases for the KeyConcept model."""
    
    def test_key_concept_creation(self, sample_key_concept):
        """Test basic key concept creation."""
        assert sample_key_concept.concept_id == "concept_1"
        assert sample_key_concept.name == "Machine Learning"
        assert sample_key_concept.description == "A subset of artificial intelligence that focuses on algorithms that can learn from data"
        assert sample_key_concept.category == "ai"
        assert sample_key_concept.confidence == 0.9
        assert sample_key_concept.source_chunks == ["chunk_1", "chunk_2"]
        assert sample_key_concept.keywords == ["AI", "algorithms", "data"]
    
    def test_key_concept_validation_confidence_range(self):
        """Test confidence validation with valid values."""
        # Valid confidence values
        concept1 = KeyConcept(
            concept_id="1",
            name="Test",
            description="Test description",
            category="general",
            confidence=0.0,
            source_chunks=["chunk_1"]
        )
        concept2 = KeyConcept(
            concept_id="2",
            name="Test",
            description="Test description",
            category="general",
            confidence=1.0,
            source_chunks=["chunk_2"]
        )
        concept3 = KeyConcept(
            concept_id="3",
            name="Test",
            description="Test description",
            category="general",
            confidence=0.5,
            source_chunks=["chunk_3"]
        )
        
        assert concept1.confidence == 0.0
        assert concept2.confidence == 1.0
        assert concept3.confidence == 0.5
    
    def test_key_concept_validation_invalid_confidence(self):
        """Test confidence validation with invalid values."""
        with pytest.raises(ValueError):
            KeyConcept(
                concept_id="1",
                name="Test",
                description="Test description",
                category="general",
                confidence=-0.1,
                source_chunks=["chunk_1"]
            )
        
        with pytest.raises(ValueError):
            KeyConcept(
                concept_id="1",
                name="Test",
                description="Test description",
                category="general",
                confidence=1.1,
                source_chunks=["chunk_1"]
            )
    
    def test_key_concept_model_dump(self, sample_key_concept):
        """Test model_dump method."""
        dumped_data = sample_key_concept.model_dump()
        assert dumped_data["concept_id"] == "concept_1"
        assert dumped_data["name"] == "Machine Learning"
        assert dumped_data["confidence"] == 0.9
        assert "created_at" in dumped_data
        assert "updated_at" in dumped_data


class TestKnowledgeRelation:
    """Test cases for the KnowledgeRelation model."""
    
    def test_knowledge_relation_creation(self, sample_knowledge_relation):
        """Test basic knowledge relation creation."""
        assert sample_knowledge_relation.relation_id == "rel_1"
        assert sample_knowledge_relation.source_concept == "concept_1"
        assert sample_knowledge_relation.target_concept == "taget_concept_2"
        assert sample_knowledge_relation.relation_type == "hierarchical"
        assert sample_knowledge_relation.strength == 0.8
        assert sample_knowledge_relation.description == "Deep learning is a subset of machine learning"
        assert len(sample_knowledge_relation.evidence) == 2
    
    def test_knowledge_relation_validation_strength_range(self):
        """Test strength validation with valid values."""
        relation1 = KnowledgeRelation(
            relation_id="rel_1",
            source_concept="c1",
            target_concept="c2",
            relation_type="causal",
            strength=0.0
        )
        relation2 = KnowledgeRelation(
            relation_id="rel_2",
            source_concept="c1",
            target_concept="c2",
            relation_type="causal",
            strength=1.0
        )
        relation3 = KnowledgeRelation(
            relation_id="rel_3",
            source_concept="c1",
            target_concept="c2",
            relation_type="causal",
            strength=0.5
        )
        
        assert relation1.strength == 0.0
        assert relation2.strength == 1.0
        assert relation3.strength == 0.5
    
    def test_knowledge_relation_validation_invalid_strength(self):
        """Test strength validation with invalid values."""
        with pytest.raises(ValueError):
            KnowledgeRelation(
                relation_id="rel_1",
                source_concept="c1",
                target_concept="c2",
                relation_type="causal",
                strength=-0.1
            )
        
        with pytest.raises(ValueError):
            KnowledgeRelation(
                relation_id="rel_1",
                source_concept="c1",
                target_concept="c2",
                relation_type="causal",
                strength=1.1
            )
    
    def test_knowledge_relation_model_dump(self, sample_knowledge_relation):
        """Test model_dump method."""
        dumped_data = sample_knowledge_relation.model_dump()
        assert dumped_data["relation_id"] == "rel_1"
        assert dumped_data["source_concept"] == "concept_1"
        assert dumped_data["strength"] == 0.8
        assert "created_at" in dumped_data


class TestDocumentKnowledge:
    """Test cases for the DocumentKnowledge model."""
    
    def test_document_knowledge_creation(self, sample_document_knowledge):
        """Test basic document knowledge creation."""
        assert sample_document_knowledge.document_id == "doc_1"
        assert sample_document_knowledge.title == "AI Fundamentals"
        assert len(sample_document_knowledge.key_concepts) == 2
        assert len(sample_document_knowledge.knowledge_relations) == 1
        assert sample_document_knowledge.main_themes == ["artificial intelligence", "machine learning"]
        assert sample_document_knowledge.quality_score == 0.85
    
    def test_get_concept_by_id(self, sample_document_knowledge):
        """Test retrieving a concept by its ID."""
        concept = sample_document_knowledge.get_concept_by_id("concept_1")
        assert concept is not None
        assert concept.name == "Machine Learning"
        
        # Test non-existent concept
        concept = sample_document_knowledge.get_concept_by_id("non_existent")
        assert concept is None
    
    def test_get_relations_for_concept(self, sample_document_knowledge):
        """Test retrieving relations for a specific concept."""
        relations = sample_document_knowledge.get_relations_for_concept("concept_1")
        assert len(relations) == 1
        assert relations[0].relation_id == "rel_1"
        
        # Test non-existent concept
        relations = sample_document_knowledge.get_relations_for_concept("non_existent")
        assert len(relations) == 0
    
    def test_get_concept_network(self, sample_document_knowledge):
        """Test getting concept network structure."""
        network = sample_document_knowledge.get_concept_network("concept_1")
        
        assert "concept" in network
        assert "incoming_relations" in network
        assert "outgoing_relations" in network
        assert network["concept"].concept_id == "concept_1"
    
    def test_document_knowledge_validation_quality_score(self):
        """Test quality score validation."""
        # Valid quality score
        doc_knowledge = DocumentKnowledge(
            document_id="doc_1",
            title="Test Document",
            summary="Test summary",
            quality_score=0.5
        )
        assert doc_knowledge.quality_score == 0.5
        
        # Test invalid quality score
        with pytest.raises(ValueError):
            DocumentKnowledge(
                document_id="doc_1",
                title="Test Document",
                summary="Test summary",
                quality_score=1.5
            )
    
    def test_document_knowledge_model_dump(self, sample_document_knowledge):
        """Test model_dump method."""
        dumped_data = sample_document_knowledge.model_dump()
        assert dumped_data["document_id"] == "doc_1"
        assert dumped_data["title"] == "AI Fundamentals"
        assert len(dumped_data["key_concepts"]) == 2
        assert "created_at" in dumped_data
        assert "updated_at" in dumped_data


class TestSubjectKnowledge:
    """Test cases for the SubjectKnowledge model."""
    
    def test_subject_knowledge_creation(self, sample_subject_knowledge):
        """Test basic subject knowledge creation."""
        assert sample_subject_knowledge.subject_id == "subject_ai"
        assert sample_subject_knowledge.name == "Artificial Intelligence"
        assert len(sample_subject_knowledge.core_concepts) == 2
        assert len(sample_subject_knowledge.knowledge_relations) == 1
        assert sample_subject_knowledge.document_sources == ["doc_1", "doc_2"]
        assert sample_subject_knowledge.expertise_level == "intermediate"
        assert sample_subject_knowledge.quality_score == 0.9
    
    def test_get_concept_by_id(self, sample_subject_knowledge):
        """Test retrieving a concept by its ID."""
        concept = sample_subject_knowledge.get_concept_by_id("concept_1")
        assert concept is not None
        assert concept.name == "Machine Learning"
        
        # Test non-existent concept
        concept = sample_subject_knowledge.get_concept_by_id("non_existent")
        assert concept is None
    
    def test_get_related_concepts(self, sample_subject_knowledge):
        """Test getting related concepts."""
        related_concepts = sample_subject_knowledge.get_related_concepts("concept_1", max_depth=1)
        assert isinstance(related_concepts, set)
        
        # Should include the target concept from the relation
        assert "concept_2" in related_concepts or "taget_concept_2" in related_concepts
    
    def test_subject_knowledge_validation_expertise_level(self):
        """Test expertise level validation."""
        # Valid expertise levels
        valid_levels = ["beginner", "intermediate", "advanced"]
        for level in valid_levels:
            subject = SubjectKnowledge(
                subject_id="test",
                name="Test Subject",
                description="Test description",
                expertise_level=level,
                quality_score=0.8
            )
            assert subject.expertise_level == level
    
    def test_subject_knowledge_validation_quality_score(self):
        """Test quality score validation."""
        # Valid quality score
        subject = SubjectKnowledge(
            subject_id="test",
            name="Test Subject",
            description="Test description",
            quality_score=0.7
        )
        assert subject.quality_score == 0.7
        
        # Test invalid quality score
        with pytest.raises(ValueError):
            SubjectKnowledge(
                subject_id="test",
                name="Test Subject",
                description="Test description",
                quality_score=1.5
            )
    
    def test_subject_knowledge_model_dump(self, sample_subject_knowledge):
        """Test model_dump method."""
        dumped_data = sample_subject_knowledge.model_dump()
        assert dumped_data["subject_id"] == "subject_ai"
        assert dumped_data["name"] == "Artificial Intelligence"
        assert len(dumped_data["core_concepts"]) == 2
        assert "created_at" in dumped_data
        assert "updated_at" in dumped_data


class TestModelIntegration:
    """Integration tests for model interactions."""
    
    def test_document_knowledge_with_concepts_and_relations(self):
        """Test DocumentKnowledge with concepts and relations."""
        concept1 = KeyConcept(
            concept_id="c1",
            name="Concept 1",
            description="Description 1",
            category="test",
            confidence=0.8,
            source_chunks=["chunk_1"]
        )
        
        concept2 = KeyConcept(
            concept_id="c2",
            name="Concept 2",
            description="Description 2",
            category="test",
            confidence=0.7,
            source_chunks=["chunk_2"]
        )
        
        relation = KnowledgeRelation(
            relation_id="r1",
            source_concept="c1",
            target_concept="c2",
            relation_type="causal",
            strength=0.6
        )
        
        doc_knowledge = DocumentKnowledge(
            document_id="doc_test",
            title="Test Document",
            summary="Test summary",
            key_concepts=[concept1, concept2],
            knowledge_relations=[relation],
            quality_score=0.8
        )
        
        # Test concept retrieval
        retrieved_concept = doc_knowledge.get_concept_by_id("c1")
        assert retrieved_concept.name == "Concept 1"
        
        # Test relation retrieval
        relations = doc_knowledge.get_relations_for_concept("c1")
        assert len(relations) == 1
        assert relations[0].relation_id == "r1"
    
    def test_subject_knowledge_with_hierarchy(self):
        """Test SubjectKnowledge with knowledge hierarchy."""
        concept1 = KeyConcept(
            concept_id="sc1",
            name="Subject Concept 1",
            description="Subject concept description",
            category="subject",
            confidence=0.9,
            source_chunks=["chunk_1"]
        )
        
        subject = SubjectKnowledge(
            subject_id="subject_test",
            name="Test Subject",
            description="Test subject description",
            core_concepts=[concept1],
            knowledge_hierarchy={
                "levels": {0: ["sc1"]},
                "root_concepts": ["sc1"],
                "leaf_concepts": ["sc1"]
            },
            quality_score=0.8
        )
        
        # Test concept retrieval
        retrieved_concept = subject.get_concept_by_id("sc1")
        assert retrieved_concept.name == "Subject Concept 1"
        
        # Test hierarchy access
        assert subject.knowledge_hierarchy["levels"][0] == ["sc1"]
        assert subject.knowledge_hierarchy["root_concepts"] == ["sc1"]
    
    def test_model_serialization_roundtrip(self, sample_document_knowledge):
        """Test model serialization and deserialization."""
        # Serialize to dict
        data = sample_document_knowledge.model_dump()
        
        # Deserialize from dict
        recreated = DocumentKnowledge(**data)
        
        # Compare key fields
        assert recreated.document_id == sample_document_knowledge.document_id
        assert recreated.title == sample_document_knowledge.title
        assert recreated.summary == sample_document_knowledge.summary
        assert len(recreated.key_concepts) == len(sample_document_knowledge.key_concepts)
        assert recreated.quality_score == sample_document_knowledge.quality_score
