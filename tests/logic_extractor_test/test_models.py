"""
Tests for the logic_extractor models module.
"""

import pytest
from logic_extractor.models import Claim, LogicalRelation, LogicExtractionSchemaLiteChunk


class TestClaim:
    """Test cases for the Claim model."""
    
    def test_claim_creation(self, sample_claim):
        """Test basic claim creation."""
        assert sample_claim.id == "claim_1"
        assert sample_claim.statement == "The Transformer model uses self-attention mechanism"
        assert sample_claim.type == "factual"
        assert sample_claim.confidence == 0.9
        assert sample_claim.derived_from is None
    
    def test_claim_with_derivation(self, sample_claim_with_derivation):
        """Test claim with derivation."""
        assert sample_claim_with_derivation.derived_from == ["claim_1"]
    
    def test_claim_validation_confidence_range(self):
        """Test claim confidence validation."""
        # Valid confidence values
        claim1 = Claim(id="1", statement="Test", type="factual", confidence=0.0)
        claim2 = Claim(id="2", statement="Test", type="factual", confidence=1.0)
        claim3 = Claim(id="3", statement="Test", type="factual", confidence=0.5)
        
        assert claim1.confidence == 0.0
        assert claim2.confidence == 1.0
        assert claim3.confidence == 0.5
    
    def test_claim_validation_invalid_confidence(self):
        """Test claim confidence validation with invalid values."""
        with pytest.raises(ValueError):
            Claim(id="1", statement="Test", type="factual", confidence=-0.1)
        
        with pytest.raises(ValueError):
            Claim(id="1", statement="Test", type="factual", confidence=1.1)
    
    def test_claim_type_validation(self):
        """Test claim type validation."""
        valid_types = ["factual", "inferential", "speculative", "assumptive"]
        for claim_type in valid_types:
            claim = Claim(id="1", statement="Test", type=claim_type)
            assert claim.type == claim_type
    
    def test_claim_model_dump(self, sample_claim):
        """Test claim model_dump method."""
        data = sample_claim.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == "claim_1"
        assert data["statement"] == "The Transformer model uses self-attention mechanism"
        assert data["type"] == "factual"
        assert data["confidence"] == 0.9


class TestLogicalRelation:
    """Test cases for the LogicalRelation model."""
    
    def test_logical_relation_creation(self, sample_logical_relation):
        """Test basic logical relation creation."""
        assert sample_logical_relation.premise == "claim_1"
        assert sample_logical_relation.conclusion == "claim_2"
        assert sample_logical_relation.relation_type == "supportive"
        assert sample_logical_relation.certainty == 0.8
    
    def test_logical_relation_validation_certainty_range(self):
        """Test logical relation certainty validation."""
        # Valid certainty values
        relation1 = LogicalRelation(
            premise="p1", conclusion="c1", relation_type="supportive", certainty=0.0
        )
        relation2 = LogicalRelation(
            premise="p1", conclusion="c1", relation_type="supportive", certainty=1.0
        )
        relation3 = LogicalRelation(
            premise="p1", conclusion="c1", relation_type="supportive", certainty=0.5
        )
        
        assert relation1.certainty == 0.0
        assert relation2.certainty == 1.0
        assert relation3.certainty == 0.5
    
    def test_logical_relation_validation_invalid_certainty(self):
        """Test logical relation certainty validation with invalid values."""
        with pytest.raises(ValueError):
            LogicalRelation(
                premise="p1", conclusion="c1", relation_type="supportive", certainty=-0.1
            )
        
        with pytest.raises(ValueError):
            LogicalRelation(
                premise="p1", conclusion="c1", relation_type="supportive", certainty=1.1
            )
    
    def test_logical_relation_type_validation(self):
        """Test logical relation type validation."""
        valid_types = ["causal", "inferential", "correlative", "contradictory", "supportive"]
        for relation_type in valid_types:
            relation = LogicalRelation(
                premise="p1", conclusion="c1", relation_type=relation_type
            )
            assert relation.relation_type == relation_type
    
    def test_logical_relation_model_dump(self, sample_logical_relation):
        """Test logical relation model_dump method."""
        data = sample_logical_relation.model_dump()
        assert isinstance(data, dict)
        assert data["premise"] == "claim_1"
        assert data["conclusion"] == "claim_2"
        assert data["relation_type"] == "supportive"
        assert data["certainty"] == 0.8


class TestLogicExtractionSchemaLiteChunk:
    """Test cases for the LogicExtractionSchemaLiteChunk model."""
    
    def test_logic_extraction_creation(self, sample_logic_extraction):
        """Test basic logic extraction creation."""
        assert sample_logic_extraction.chunk_id == "test_chunk_1"
        assert len(sample_logic_extraction.claims) == 2
        assert len(sample_logic_extraction.logical_relations) == 1
        assert len(sample_logic_extraction.assumptions) == 1
        assert len(sample_logic_extraction.constraints) == 1
        assert len(sample_logic_extraction.open_questions) == 1
    
    def test_logic_extraction_default_values(self):
        """Test logic extraction with default values."""
        extraction = LogicExtractionSchemaLiteChunk(chunk_id="test")
        
        assert extraction.chunk_id == "test"
        assert extraction.claims == []
        assert extraction.logical_relations == []
        assert extraction.assumptions == []
        assert extraction.constraints == []
        assert extraction.open_questions == []
    
    def test_get_claim_by_id(self, sample_logic_extraction):
        """Test getting claim by ID."""
        claim = sample_logic_extraction.get_claim_by_id("claim_1")
        assert claim is not None
        assert claim.id == "claim_1"
        assert claim.statement == "The Transformer model uses self-attention mechanism"
        
        # Test non-existent claim
        claim = sample_logic_extraction.get_claim_by_id("non_existent")
        assert claim is None
    
    def test_get_relations_by_premise(self, sample_logic_extraction):
        """Test getting relations by premise."""
        relations = sample_logic_extraction.get_relations_by_premise("claim_1")
        assert len(relations) == 1
        assert relations[0].premise == "claim_1"
        assert relations[0].conclusion == "claim_2"
        
        # Test non-existent premise
        relations = sample_logic_extraction.get_relations_by_premise("non_existent")
        assert len(relations) == 0
    
    def test_get_relations_by_conclusion(self, sample_logic_extraction):
        """Test getting relations by conclusion."""
        relations = sample_logic_extraction.get_relations_by_conclusion("claim_2")
        assert len(relations) == 1
        assert relations[0].premise == "claim_1"
        assert relations[0].conclusion == "claim_2"
        
        # Test non-existent conclusion
        relations = sample_logic_extraction.get_relations_by_conclusion("non_existent")
        assert len(relations) == 0
    
    def test_get_claim_network(self, sample_logic_extraction):
        """Test getting claim network structure."""
        network = sample_logic_extraction.get_claim_network()
        
        assert "claim_1" in network
        assert "claim_2" in network
        
        # Check claim_1 network
        claim_1_network = network["claim_1"]
        assert claim_1_network["claim"].id == "claim_1"
        assert len(claim_1_network["premises"]) == 1  # claim_1 is a premise for claim_2
        assert len(claim_1_network["conclusions"]) == 0  # claim_1 is premise for claim_2
        
        # Check claim_2 network
        claim_2_network = network["claim_2"]
        assert claim_2_network["claim"].id == "claim_2"
        assert len(claim_2_network["premises"]) == 0  # claim_2 has no premises
        assert len(claim_2_network["conclusions"]) == 1  # claim_2 is a conclusion for claim_1
    
    def test_validate_claim_references_valid(self, sample_logic_extraction):
        """Test claim reference validation with valid references."""
        errors = sample_logic_extraction.validate_claim_references()
        assert len(errors) == 0
    
    def test_validate_claim_references_invalid(self):
        """Test claim reference validation with invalid references."""
        claims = [
            Claim(id="claim_1", statement="Test 1", type="factual")
        ]
        
        relations = [
            LogicalRelation(
                premise="claim_1",
                conclusion="non_existent",
                relation_type="supportive"
            ),
            LogicalRelation(
                premise="non_existent",
                conclusion="claim_1",
                relation_type="supportive"
            )
        ]
        
        extraction = LogicExtractionSchemaLiteChunk(
            chunk_id="test",
            claims=claims,
            logical_relations=relations
        )
        
        errors = extraction.validate_claim_references()
        assert len(errors) == 2
        assert any("non_existent" in error for error in errors)
    
    def test_logic_extraction_model_dump(self, sample_logic_extraction):
        """Test logic extraction model_dump method."""
        data = sample_logic_extraction.model_dump()
        assert isinstance(data, dict)
        assert data["chunk_id"] == "test_chunk_1"
        assert len(data["claims"]) == 2
        assert len(data["logical_relations"]) == 1
        assert len(data["assumptions"]) == 1
        assert len(data["constraints"]) == 1
        assert len(data["open_questions"]) == 1
    
    def test_logic_extraction_equality(self, sample_logic_extraction):
        """Test logic extraction equality comparison."""
        # Create identical extraction
        identical = LogicExtractionSchemaLiteChunk(
            chunk_id="test_chunk_1",
            claims=sample_logic_extraction.claims.copy(),
            logical_relations=sample_logic_extraction.logical_relations.copy(),
            assumptions=sample_logic_extraction.assumptions.copy(),
            constraints=sample_logic_extraction.constraints.copy(),
            open_questions=sample_logic_extraction.open_questions.copy()
        )
        
        # Note: Direct equality comparison may not work due to object identity
        # Instead, we can compare the model_dump results
        assert sample_logic_extraction.model_dump() == identical.model_dump()
    
    def test_logic_extraction_with_none_values(self):
        """Test logic extraction with None values for optional fields."""
        extraction = LogicExtractionSchemaLiteChunk(
            chunk_id="test",
            claims=[],
            logical_relations=[],
            assumptions=None,
            constraints=None,
            open_questions=None
        )
        
        # When None is passed explicitly, it should remain None
        # The default_factory only applies when the field is not provided
        assert extraction.assumptions is None
        assert extraction.constraints is None
        assert extraction.open_questions is None
    
    def test_logic_extraction_with_empty_lists(self):
        """Test logic extraction with empty lists."""
        extraction = LogicExtractionSchemaLiteChunk(
            chunk_id="test",
            claims=[],
            logical_relations=[],
            assumptions=[],
            constraints=[],
            open_questions=[]
        )
        
        assert extraction.claims == []
        assert extraction.logical_relations == []
        assert extraction.assumptions == []
        assert extraction.constraints == []
        assert extraction.open_questions == []
