"""
Unit tests for cluster integration.

This module contains tests for the ClusterBasedSubjectConsolidator
and IntegratedConsolidationPipeline classes.
"""

import pytest
from unittest.mock import Mock, patch

from consolidation.cluster_integration import (
    ClusterBasedSubjectConsolidator,
    IntegratedConsolidationPipeline
)
from consolidation.models import DocumentKnowledge, SubjectKnowledge, KeyConcept, KnowledgeRelation
from consolidation.config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from consolidation.exceptions import SubjectConsolidationError, ConsolidationError


class TestClusterBasedSubjectConsolidator:
    """Test cases for the ClusterBasedSubjectConsolidator class."""
    
    @pytest.fixture
    def sample_clustering_result(self):
        """Create sample clustering result for testing."""
        from clustering.models import ClusteringResult, ClusterInfo, DocumentClusterAssignment
        
        cluster_info = ClusterInfo(
            cluster_id=1,
            name="AI Cluster",
            topic_words=["artificial intelligence", "machine learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        assignments = [
            DocumentClusterAssignment(
                document_id="doc_1",
                cluster_id=1,
                confidence=0.9,
                distance_to_center=0.2
            ),
            DocumentClusterAssignment(
                document_id="doc_2",
                cluster_id=1,
                confidence=0.8,
                distance_to_center=0.3
            )
        ]
        
        return ClusteringResult(
            clusters=[cluster_info],
            assignments=assignments,
            total_documents=2,
            num_clusters=1,
            metadata={"method": "BERTopic", "num_topics": 1}
        )
    
    @pytest.fixture
    def sample_document_knowledge_map(self):
        """Create sample document knowledge map for testing."""
        doc1 = DocumentKnowledge(
            document_id="doc_1",
            title="AI Document 1",
            summary="First AI document",
            key_concepts=[
                KeyConcept(concept_id="c1", name="ML", description="Machine Learning", category="ai", confidence=0.8, source_chunks=["chunk_1"])
            ],
            knowledge_relations=[],
            main_themes=["AI", "ML"],
            knowledge_graph={},
            quality_score=0.85
        )
        
        doc2 = DocumentKnowledge(
            document_id="doc_2",
            title="AI Document 2",
            summary="Second AI document",
            key_concepts=[
                KeyConcept(concept_id="c2", name="AI", description="Artificial Intelligence", category="ai", confidence=0.9, source_chunks=["chunk_2"])
            ],
            knowledge_relations=[],
            main_themes=["AI"],
            knowledge_graph={},
            quality_score=0.8
        )
        
        return {"doc_1": doc1, "doc_2": doc2}
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        assert isinstance(consolidator.config, ConsolidationConfig)
        assert consolidator.config.model_name == "gpt-4o-mini"
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ConsolidationConfig(model_name="gpt-4o", temperature=0.2)
        consolidator = ClusterBasedSubjectConsolidator(config)
        
        assert consolidator.config == config
    
    def test_consolidate_subjects_from_clusters_success(self, sample_clustering_result, sample_document_knowledge_map):
        """Test successful subject consolidation from clusters."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        # Mock the subject consolidator
        with patch.object(consolidator.subject_consolidator, 'consolidate_subject') as mock_consolidate:
            mock_consolidate.return_value = SubjectKnowledge(
                subject_id="subject_cluster_1",
                name="AI Cluster",
                description="Generated description",
                core_concepts=[],
                knowledge_relations=[],
                document_sources=["doc_1", "doc_2"],
                knowledge_hierarchy={},
                expertise_level="intermediate",
                domain_tags=["AI"],
                quality_score=0.8
            )
            
            result = consolidator.consolidate_subjects_from_clusters(
                sample_clustering_result, sample_document_knowledge_map
            )
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], SubjectKnowledge)
            assert result[0].subject_id == "subject_cluster_1"
    
    def test_consolidate_subjects_from_clusters_no_documents(self, sample_clustering_result):
        """Test subject consolidation with no documents for cluster."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        # Empty document knowledge map
        document_knowledge_map = {}
        
        result = consolidator.consolidate_subjects_from_clusters(
            sample_clustering_result, document_knowledge_map
        )
        
        assert isinstance(result, list)
        assert len(result) == 0  # Should have no subjects due to missing documents
    
    def test_consolidate_subjects_from_clusters_failure(self, sample_clustering_result, sample_document_knowledge_map):
        """Test subject consolidation failure."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        # Mock the subject consolidator to raise an exception
        with patch.object(consolidator.subject_consolidator, 'consolidate_subject', side_effect=Exception("Consolidation error")):
            with pytest.raises(SubjectConsolidationError, match="Failed to consolidate subjects from clusters"):
                consolidator.consolidate_subjects_from_clusters(
                    sample_clustering_result, sample_document_knowledge_map
                )
    
    def test_get_documents_for_cluster(self, sample_clustering_result, sample_document_knowledge_map):
        """Test getting documents for a specific cluster."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        cluster = sample_clustering_result.clusters[0]
        documents = consolidator._get_documents_for_cluster(
            cluster, sample_clustering_result.assignments, sample_document_knowledge_map
        )
        
        assert isinstance(documents, list)
        assert len(documents) == 2  # Should have 2 documents for this cluster
        assert all(isinstance(doc, DocumentKnowledge) for doc in documents)
    
    def test_get_documents_for_cluster_missing_documents(self, sample_clustering_result):
        """Test getting documents for cluster with missing documents."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        cluster = sample_clustering_result.clusters[0]
        # Document knowledge map with missing documents
        document_knowledge_map = {"doc_1": Mock()}  # Missing doc_2
        
        documents = consolidator._get_documents_for_cluster(
            cluster, sample_clustering_result.assignments, document_knowledge_map
        )
        
        assert isinstance(documents, list)
        assert len(documents) == 1  # Should have only 1 document
    
    def test_get_documents_for_cluster_exception_handling(self, sample_clustering_result, sample_document_knowledge_map):
        """Test getting documents for cluster with exception handling."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        cluster = sample_clustering_result.clusters[0]
        # Create assignments that might cause issues
        problematic_assignments = [Mock()]  # Mock object without expected attributes
        
        documents = consolidator._get_documents_for_cluster(
            cluster, problematic_assignments, sample_document_knowledge_map
        )
        
        # Should return empty list on exception
        assert documents == []
    
    def test_consolidate_single_subject_success(self, sample_document_knowledge_map):
        """Test successful consolidation of a single subject."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="AI Cluster",
            topic_words=["artificial intelligence", "machine learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        cluster_documents = list(sample_document_knowledge_map.values())
        
        # Mock the subject consolidator
        with patch.object(consolidator.subject_consolidator, 'consolidate_subject') as mock_consolidate:
            mock_consolidate.return_value = SubjectKnowledge(
                subject_id="subject_cluster_1",
                name="AI Cluster",
                description="Generated description",
                core_concepts=[],
                knowledge_relations=[],
                document_sources=["doc_1", "doc_2"],
                knowledge_hierarchy={},
                expertise_level="intermediate",
                domain_tags=["AI"],
                quality_score=0.8
            )
            
            result = consolidator._consolidate_single_subject(cluster, cluster_documents)
            
            assert isinstance(result, SubjectKnowledge)
            assert result.subject_id == "subject_cluster_1"
            assert "cluster_id" in result.consolidation_metadata
            assert result.consolidation_metadata["cluster_id"] == 1
    
    def test_consolidate_single_subject_failure(self, sample_document_knowledge_map):
        """Test consolidation of a single subject failure."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="AI Cluster",
            topic_words=["artificial intelligence", "machine learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        cluster_documents = list(sample_document_knowledge_map.values())
        
        # Mock the subject consolidator to raise an exception
        with patch.object(consolidator.subject_consolidator, 'consolidate_subject', side_effect=Exception("Consolidation error")):
            result = consolidator._consolidate_single_subject(cluster, cluster_documents)
            
            # Should return None on failure
            assert result is None
    
    def test_generate_subject_name_from_cluster_name(self, sample_document_knowledge_map):
        """Test subject name generation from cluster name."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="AI and Machine Learning",
            topic_words=["artificial intelligence", "machine learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        cluster_documents = list(sample_document_knowledge_map.values())
        
        subject_name = consolidator._generate_subject_name(cluster, cluster_documents)
        
        assert subject_name == "AI and Machine Learning"  # Should use cluster name
    
    def test_generate_subject_name_from_topic_words(self, sample_document_knowledge_map):
        """Test subject name generation from topic words."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="Cluster 1",  # Generic name
            topic_words=["artificial intelligence", "machine learning", "deep learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        cluster_documents = list(sample_document_knowledge_map.values())
        
        subject_name = consolidator._generate_subject_name(cluster, cluster_documents)
        
        assert "Artificial Intelligence" in subject_name  # Should use topic words
    
    def test_generate_subject_name_from_document_themes(self, sample_document_knowledge_map):
        """Test subject name generation from document themes."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="Cluster 1",  # Generic name
            topic_words=[],  # No topic words
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        cluster_documents = list(sample_document_knowledge_map.values())
        
        subject_name = consolidator._generate_subject_name(cluster, cluster_documents)
        
        assert "AI" in subject_name or "Subject 1" in subject_name  # Should use themes or fallback
    
    def test_generate_subject_name_exception_handling(self, sample_document_knowledge_map):
        """Test subject name generation with exception handling."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="Cluster 1",
            topic_words=["artificial intelligence", "machine learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        # Create documents that might cause issues
        problematic_documents = [Mock()]  # Mock object without expected attributes
        
        subject_name = consolidator._generate_subject_name(cluster, problematic_documents)
        
        # Should return fallback name on exception
        assert subject_name == "Subject 1"
    
    def test_generate_subject_description(self, sample_document_knowledge_map):
        """Test subject description generation."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="AI Cluster",
            topic_words=["artificial intelligence", "machine learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        cluster_documents = list(sample_document_knowledge_map.values())
        
        description = consolidator._generate_subject_description(cluster, cluster_documents)
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "artificial intelligence" in description.lower()
    
    def test_generate_subject_description_exception_handling(self, sample_document_knowledge_map):
        """Test subject description generation with exception handling."""
        consolidator = ClusterBasedSubjectConsolidator()
        
        from clustering.models import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=1,
            name="AI Cluster",
            topic_words=["artificial intelligence", "machine learning"],
            document_count=2,
            coherence_score=0.8,
            silhouette_score=0.7
        )
        
        # Create documents that might cause issues
        problematic_documents = [Mock()]  # Mock object without expected attributes
        
        description = consolidator._generate_subject_description(cluster, problematic_documents)
        
        # Should return fallback description on exception
        assert isinstance(description, str)
        assert "cluster 1" in description.lower()


class TestIntegratedConsolidationPipeline:
    """Test cases for the IntegratedConsolidationPipeline class."""
    
    @pytest.fixture
    def sample_documents_and_ids(self):
        """Create sample documents and IDs for testing."""
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing combines linguistics with ML."
        ]
        
        document_ids = ["doc_1", "doc_2", "doc_3"]
        
        return documents, document_ids
    
    @pytest.fixture
    def sample_chunks_map(self):
        """Create sample chunks map for testing."""
        return {
            "doc_1": [{"content": "Machine learning is a subset of artificial intelligence.", "metadata": {"page": 1}}],
            "doc_2": [{"content": "Deep learning uses neural networks with multiple layers.", "metadata": {"page": 1}}],
            "doc_3": [{"content": "Natural language processing combines linguistics with ML.", "metadata": {"page": 1}}]
        }
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        pipeline = IntegratedConsolidationPipeline()
        
        assert isinstance(pipeline.consolidation_config, ConsolidationConfig)
        assert pipeline.consolidation_config.model_name == "gpt-4o-mini"
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ConsolidationConfig(model_name="gpt-4o", temperature=0.2)
        pipeline = IntegratedConsolidationPipeline(config)
        
        assert pipeline.consolidation_config == config
    
    def test_process_documents_to_subjects_success(self, sample_documents_and_ids, sample_chunks_map):
        """Test successful document processing to subjects."""
        documents, document_ids = sample_documents_and_ids
        
        pipeline = IntegratedConsolidationPipeline()
        
        # Mock the document consolidator
        with patch('consolidation.cluster_integration.DocumentConsolidator') as mock_doc_consolidator_class:
            mock_doc_consolidator = Mock()
            mock_doc_consolidator.consolidate_document.return_value = DocumentKnowledge(
                document_id="doc_1",
                title="Test Document",
                summary="Test summary",
                key_concepts=[],
                knowledge_relations=[],
                main_themes=["AI"],
                knowledge_graph={},
                quality_score=0.8
            )
            mock_doc_consolidator_class.return_value = mock_doc_consolidator
            
            # Mock the clusterer
            with patch('consolidation.cluster_integration.DocumentClusterer') as mock_clusterer_class:
                mock_clusterer = Mock()
                mock_clustering_result = Mock()
                mock_clusterer.fit_clusters.return_value = mock_clustering_result
                mock_clusterer_class.return_value = mock_clusterer
                
                # Mock the cluster-based consolidator
                with patch.object(pipeline.cluster_based_consolidator, 'consolidate_subjects_from_clusters') as mock_consolidate:
                    mock_consolidate.return_value = [
                        SubjectKnowledge(
                            subject_id="subject_1",
                            name="AI Subject",
                            description="AI knowledge",
                            core_concepts=[],
                            knowledge_relations=[],
                            document_sources=["doc_1"],
                            knowledge_hierarchy={},
                            expertise_level="intermediate",
                            domain_tags=["AI"],
                            quality_score=0.8
                        )
                    ]
                    
                    doc_knowledge_list, subject_knowledge_list = pipeline.process_documents_to_subjects(
                        documents=documents,
                        document_ids=document_ids,
                        document_chunks_map=sample_chunks_map
                    )
                    
                    assert isinstance(doc_knowledge_list, list)
                    assert isinstance(subject_knowledge_list, list)
                    assert len(doc_knowledge_list) == 3  # Should have 3 documents
                    assert len(subject_knowledge_list) == 1  # Should have 1 subject
                    assert all(isinstance(doc, DocumentKnowledge) for doc in doc_knowledge_list)
                    assert all(isinstance(subj, SubjectKnowledge) for subj in subject_knowledge_list)
    
    def test_process_documents_to_subjects_failure(self, sample_documents_and_ids, sample_chunks_map):
        """Test document processing failure."""
        documents, document_ids = sample_documents_and_ids
        
        pipeline = IntegratedConsolidationPipeline()
        
        # Mock the document consolidator to raise an exception
        with patch('consolidation.cluster_integration.DocumentConsolidator', side_effect=Exception("Document consolidation error")):
            with pytest.raises(ConsolidationError, match="Integrated pipeline failed"):
                pipeline.process_documents_to_subjects(
                    documents=documents,
                    document_ids=document_ids,
                    document_chunks_map=sample_chunks_map
                )
    
    def test_process_documents_to_subjects_with_logic_data(self, sample_documents_and_ids, sample_chunks_map):
        """Test document processing with logic data."""
        documents, document_ids = sample_documents_and_ids
        
        # Add logic data map
        logic_data_map = {
            "doc_1": [{"claims": [], "relations": []}],
            "doc_2": [{"claims": [], "relations": []}],
            "doc_3": [{"claims": [], "relations": []}]
        }
        
        pipeline = IntegratedConsolidationPipeline()
        
        # Mock the document consolidator
        with patch('consolidation.cluster_integration.DocumentConsolidator') as mock_doc_consolidator_class:
            mock_doc_consolidator = Mock()
            mock_doc_consolidator.consolidate_document.return_value = DocumentKnowledge(
                document_id="doc_1",
                title="Test Document",
                summary="Test summary",
                key_concepts=[],
                knowledge_relations=[],
                main_themes=["AI"],
                knowledge_graph={},
                quality_score=0.8
            )
            mock_doc_consolidator_class.return_value = mock_doc_consolidator
            
            # Mock the clusterer
            with patch('consolidation.cluster_integration.DocumentClusterer') as mock_clusterer_class:
                mock_clusterer = Mock()
                mock_clustering_result = Mock()
                mock_clusterer.fit_clusters.return_value = mock_clustering_result
                mock_clusterer_class.return_value = mock_clusterer
                
                # Mock the cluster-based consolidator
                with patch.object(pipeline.cluster_based_consolidator, 'consolidate_subjects_from_clusters') as mock_consolidate:
                    mock_consolidate.return_value = []
                    
                    doc_knowledge_list, subject_knowledge_list = pipeline.process_documents_to_subjects(
                        documents=documents,
                        document_ids=document_ids,
                        document_chunks_map=sample_chunks_map,
                        chunk_logic_data_map=logic_data_map
                    )
                    
                    assert isinstance(doc_knowledge_list, list)
                    assert isinstance(subject_knowledge_list, list)
                    # Should have processed documents with logic data
                    assert len(doc_knowledge_list) >= 0
    
    def test_process_documents_to_subjects_document_consolidation_failure(self, sample_documents_and_ids, sample_chunks_map):
        """Test document processing with document consolidation failure."""
        documents, document_ids = sample_documents_and_ids
        
        pipeline = IntegratedConsolidationPipeline()
        
        # Mock the document consolidator to fail for some documents
        with patch('consolidation.cluster_integration.DocumentConsolidator') as mock_doc_consolidator_class:
            mock_doc_consolidator = Mock()
            # Make it succeed for first document, fail for others
            def side_effect(*args, **kwargs):
                if kwargs.get('document_id') == 'doc_1':
                    return DocumentKnowledge(
                        document_id="doc_1",
                        title="Test Document",
                        summary="Test summary",
                        key_concepts=[],
                        knowledge_relations=[],
                        main_themes=["AI"],
                        knowledge_graph={},
                        quality_score=0.8
                    )
                else:
                    raise Exception("Document consolidation failed")
            
            mock_doc_consolidator.consolidate_document.side_effect = side_effect
            mock_doc_consolidator_class.return_value = mock_doc_consolidator
            
            # Mock the clusterer
            with patch('consolidation.cluster_integration.DocumentClusterer') as mock_clusterer_class:
                mock_clusterer = Mock()
                mock_clustering_result = Mock()
                mock_clusterer.fit_clusters.return_value = mock_clustering_result
                mock_clusterer_class.return_value = mock_clusterer
                
                # Mock the cluster-based consolidator
                with patch.object(pipeline.cluster_based_consolidator, 'consolidate_subjects_from_clusters') as mock_consolidate:
                    mock_consolidate.return_value = []
                    
                    doc_knowledge_list, subject_knowledge_list = pipeline.process_documents_to_subjects(
                        documents=documents,
                        document_ids=document_ids,
                        document_chunks_map=sample_chunks_map
                    )
                    
                    assert isinstance(doc_knowledge_list, list)
                    assert isinstance(subject_knowledge_list, list)
                    # Should have processed only successful documents
                    assert len(doc_knowledge_list) == 1  # Only doc_1 should succeed
