"""
Subject-level knowledge consolidation.

This module implements subject-level consolidation functionality that
aggregates document knowledge into general subject knowledge.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import uuid

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import Runnable
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import numpy as np
except ImportError as e:
    raise ImportError(
        "Required consolidation dependencies not installed. "
        "Please install: pip install langchain-openai sentence-transformers scikit-learn"
    ) from e

from .models import SubjectKnowledge, KeyConcept, KnowledgeRelation, DocumentKnowledge
from .config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from .exceptions import (
    SubjectConsolidationError, ConceptExtractionError, RelationExtractionError,
    MissingAPIKeyError, KnowledgeValidationError, KnowledgeMergeError
)
from .utils import merge_related_concepts, validate_knowledge_consistency

logger = logging.getLogger(__name__)


class SubjectConsolidator:
    """
    Consolidates knowledge from multiple documents into subject-level knowledge.
    
    This class handles the process of aggregating document knowledge,
    merging related concepts, and building comprehensive subject knowledge.
    """
    
    def __init__(self, config: ConsolidationConfig = DEFAULT_CONSOLIDATION_CONFIG):
        """
        Initialize the SubjectConsolidator.
        
        Args:
            config: Configuration for consolidation parameters
        """
        self.config = config
        self.llm: Optional[Runnable] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize LLM and embedding models."""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Initialize embedding model if vector search is enabled
            if self.config.enable_vector_search:
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            logger.info(f"Initialized SubjectConsolidator with model: {self.config.model_name}")
            
        except Exception as e:
            raise MissingAPIKeyError(f"Failed to initialize models: {e}") from e
    
    def consolidate_subject(
        self,
        subject_id: str,
        subject_name: str,
        document_knowledge: List[DocumentKnowledge],
        subject_description: Optional[str] = None
    ) -> SubjectKnowledge:
        """
        Consolidate knowledge from multiple documents into subject knowledge.
        
        Args:
            subject_id: Unique identifier for the subject
            subject_name: Name of the subject
            document_knowledge: List of document knowledge to consolidate
            subject_description: Optional description of the subject
            
        Returns:
            SubjectKnowledge containing consolidated subject knowledge
            
        Raises:
            SubjectConsolidationError: If consolidation fails
        """
        try:
            logger.info(f"Starting subject consolidation for subject: {subject_id}")
            
            # Extract and merge concepts from all documents
            core_concepts = self._extract_and_merge_concepts(document_knowledge)
            
            # Extract and merge relations
            knowledge_relations = self._extract_and_merge_relations(document_knowledge, core_concepts)
            
            # Build knowledge hierarchy
            knowledge_hierarchy = self._build_knowledge_hierarchy(core_concepts, knowledge_relations)
            
            # Generate subject description if not provided
            if not subject_description:
                subject_description = self._generate_subject_description(core_concepts, document_knowledge)
            
            # Determine expertise level
            expertise_level = self._determine_expertise_level(core_concepts, knowledge_relations)
            
            # Extract domain tags
            domain_tags = self._extract_domain_tags(core_concepts, document_knowledge)
            
            # Calculate quality score
            quality_score = self._calculate_subject_quality_score(core_concepts, knowledge_relations)
            
            # Get document sources
            document_sources = [doc.document_id for doc in document_knowledge]
            
            # Create subject knowledge
            subject_knowledge = SubjectKnowledge(
                subject_id=subject_id,
                name=subject_name,
                description=subject_description,
                core_concepts=core_concepts,
                knowledge_relations=knowledge_relations,
                document_sources=document_sources,
                knowledge_hierarchy=knowledge_hierarchy,
                expertise_level=expertise_level,
                domain_tags=domain_tags,
                quality_score=quality_score,
                consolidation_metadata={
                    "num_documents": len(document_knowledge),
                    "num_concepts": len(core_concepts),
                    "num_relations": len(knowledge_relations),
                    "consolidation_timestamp": datetime.now().isoformat(),
                    "config_used": self.config.model_dump()
                }
            )
            
            # Validate knowledge if enabled
            if self.config.enable_quality_validation:
                if not validate_knowledge_consistency(subject_knowledge):
                    raise KnowledgeValidationError("Subject knowledge validation failed")
            
            logger.info(f"Successfully consolidated subject {subject_id} with {len(core_concepts)} concepts")
            return subject_knowledge
            
        except Exception as e:
            raise SubjectConsolidationError(f"Failed to consolidate subject {subject_id}: {e}") from e
    
    def _extract_and_merge_concepts(self, document_knowledge: List[DocumentKnowledge]) -> List[KeyConcept]:
        """Extract and merge concepts from all documents."""
        try:
            # Collect all concepts from documents
            all_concepts = []
            for doc in document_knowledge:
                for concept in doc.key_concepts:
                    # Add document source information
                    concept.source_chunks.append(f"doc_{doc.document_id}")
                    all_concepts.append(concept)
            
            if not all_concepts:
                return []
            
            # Merge similar concepts
            merged_concepts = self._merge_similar_concepts_across_documents(all_concepts)
            
            # Filter by confidence and limit number
            filtered_concepts = [
                concept for concept in merged_concepts
                if concept.confidence >= self.config.min_concept_confidence
            ]
            
            # Sort by confidence and limit
            filtered_concepts = sorted(
                filtered_concepts,
                key=lambda x: x.confidence,
                reverse=True
            )[:self.config.max_concepts_per_subject]
            
            return filtered_concepts
            
        except Exception as e:
            raise ConceptExtractionError(f"Failed to extract and merge concepts: {e}") from e
    
    def _merge_similar_concepts_across_documents(self, concepts: List[KeyConcept]) -> List[KeyConcept]:
        """Merge similar concepts across documents."""
        if not self.embedding_model or len(concepts) <= 1:
            return concepts
        
        try:
            # Generate embeddings for concept names and descriptions
            concept_texts = [f"{c.name} {c.description}" for c in concepts]
            embeddings = self.embedding_model.encode(concept_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find similar concepts
            merged_indices = set()
            merged_concepts = []
            
            for i, concept in enumerate(concepts):
                if i in merged_indices:
                    continue
                
                # Find similar concepts
                similar_indices = [
                    j for j in range(i + 1, len(concepts))
                    if similarity_matrix[i][j] >= self.config.concept_similarity_threshold
                    and j not in merged_indices
                ]
                
                if similar_indices:
                    # Merge with similar concepts
                    merged_concept = self._merge_concept_group_across_documents(
                        [concept] + [concepts[j] for j in similar_indices]
                    )
                    merged_concepts.append(merged_concept)
                    merged_indices.update([i] + similar_indices)
                else:
                    merged_concepts.append(concept)
            
            return merged_concepts
            
        except Exception as e:
            logger.warning(f"Failed to merge similar concepts: {e}")
            return concepts
    
    def _merge_concept_group_across_documents(self, concepts: List[KeyConcept]) -> KeyConcept:
        """Merge a group of similar concepts across documents."""
        if len(concepts) == 1:
            return concepts[0]
        
        # Use the concept with highest confidence as base
        base_concept = max(concepts, key=lambda x: x.confidence)
        
        # Merge information
        merged_keywords = list(set([kw for c in concepts for kw in c.keywords]))
        merged_source_chunks = list(set([sc for c in concepts for sc in c.source_chunks]))
        
        # Merge descriptions using LLM
        merged_description = self._merge_concept_descriptions(concepts)
        
        # Create merged concept
        merged_concept = KeyConcept(
            concept_id=f"subject_{base_concept.concept_id}",
            name=base_concept.name,
            description=merged_description,
            category=base_concept.category,
            confidence=max(c.confidence for c in concepts),
            source_chunks=merged_source_chunks,
            keywords=merged_keywords,
            created_at=base_concept.created_at,
            updated_at=datetime.now()
        )
        
        return merged_concept
    
    def _merge_concept_descriptions(self, concepts: List[KeyConcept]) -> str:
        """Merge concept descriptions using LLM."""
        try:
            if len(concepts) == 1:
                return concepts[0].description
            
            # Create prompt for merging descriptions
            descriptions = [f"Description {i+1}: {c.description}" for i, c in enumerate(concepts)]
            concept_name = concepts[0].name
            
            prompt = f"""
            Merge the following descriptions of the concept "{concept_name}" into a single, comprehensive description:
            
            {chr(10).join(descriptions)}
            
            Provide a unified description that captures all the important aspects from the individual descriptions.
            """
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to merge concept descriptions: {e}")
            return concepts[0].description
    
    def _extract_and_merge_relations(
        self, 
        document_knowledge: List[DocumentKnowledge], 
        core_concepts: List[KeyConcept]
    ) -> List[KnowledgeRelation]:
        """Extract and merge relations from all documents."""
        try:
            # Create concept ID mapping
            concept_id_map = {c.concept_id: c for c in core_concepts}
            
            # Collect all relations from documents
            all_relations = []
            for doc in document_knowledge:
                for relation in doc.knowledge_relations:
                    # Check if both concepts are in core concepts
                    if (relation.source_concept in concept_id_map and 
                        relation.target_concept in concept_id_map):
                        all_relations.append(relation)
            
            # Merge duplicate relations
            merged_relations = self._merge_duplicate_relations(all_relations)
            
            # Filter by strength threshold
            filtered_relations = [
                rel for rel in merged_relations
                if rel.strength >= self.config.relation_strength_threshold
            ]
            
            return filtered_relations
            
        except Exception as e:
            raise RelationExtractionError(f"Failed to extract and merge relations: {e}") from e
    
    def _merge_duplicate_relations(self, relations: List[KnowledgeRelation]) -> List[KnowledgeRelation]:
        """Merge duplicate relations."""
        # Group relations by source-target pairs
        relation_groups = {}
        for relation in relations:
            key = (relation.source_concept, relation.target_concept)
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(relation)
        
        # Merge relations in each group
        merged_relations = []
        for (source, target), group in relation_groups.items():
            if len(group) == 1:
                merged_relations.append(group[0])
            else:
                # Merge relations
                merged_relation = self._merge_relation_group_across_documents(group)
                merged_relations.append(merged_relation)
        
        return merged_relations
    
    def _merge_relation_group_across_documents(self, relations: List[KnowledgeRelation]) -> KnowledgeRelation:
        """Merge a group of relations across documents."""
        if len(relations) == 1:
            return relations[0]
        
        # Use the relation with highest strength as base
        base_relation = max(relations, key=lambda x: x.strength)
        
        # Merge information
        merged_evidence = list(set([ev for r in relations for ev in r.evidence]))
        merged_description = self._merge_relation_descriptions(relations)
        
        # Create merged relation
        merged_relation = KnowledgeRelation(
            relation_id=f"subject_{base_relation.relation_id}",
            source_concept=base_relation.source_concept,
            target_concept=base_relation.target_concept,
            relation_type=base_relation.relation_type,
            strength=max(r.strength for r in relations),
            description=merged_description,
            evidence=merged_evidence,
            created_at=base_relation.created_at
        )
        
        return merged_relation
    
    def _merge_relation_descriptions(self, relations: List[KnowledgeRelation]) -> str:
        """Merge relation descriptions using LLM."""
        try:
            if len(relations) == 1:
                return relations[0].description
            
            # Create prompt for merging descriptions
            descriptions = [f"Description {i+1}: {r.description}" for i, r in enumerate(relations)]
            relation_type = relations[0].relation_type
            
            prompt = f"""
            Merge the following descriptions of a {relation_type} relationship into a single, comprehensive description:
            
            {chr(10).join(descriptions)}
            
            Provide a unified description that captures all the important aspects from the individual descriptions.
            """
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to merge relation descriptions: {e}")
            return relations[0].description
    
    def _build_knowledge_hierarchy(
        self, 
        concepts: List[KeyConcept], 
        relations: List[KnowledgeRelation]
    ) -> Dict[str, Any]:
        """Build a hierarchical structure of knowledge."""
        try:
            # Create concept network
            concept_network = {}
            for concept in concepts:
                concept_network[concept.concept_id] = {
                    "concept": concept,
                    "children": [],
                    "parents": [],
                    "level": 0
                }
            
            # Add relations to network
            for relation in relations:
                if (relation.source_concept in concept_network and 
                    relation.target_concept in concept_network):
                    
                    # Add hierarchical relations
                    if relation.relation_type in ["hierarchical", "part_of", "subtype_of"]:
                        concept_network[relation.source_concept]["children"].append(relation.target_concept)
                        concept_network[relation.target_concept]["parents"].append(relation.source_concept)
            
            # Calculate hierarchy levels
            self._calculate_hierarchy_levels(concept_network)
            
            # Build hierarchy structure
            hierarchy = {
                "levels": {},
                "root_concepts": [],
                "leaf_concepts": []
            }
            
            # Group concepts by level
            for concept_id, network_info in concept_network.items():
                level = network_info["level"]
                if level not in hierarchy["levels"]:
                    hierarchy["levels"][level] = []
                hierarchy["levels"][level].append(concept_id)
                
                # Identify root and leaf concepts
                if not network_info["parents"]:
                    hierarchy["root_concepts"].append(concept_id)
                if not network_info["children"]:
                    hierarchy["leaf_concepts"].append(concept_id)
            
            return hierarchy
            
        except Exception as e:
            logger.warning(f"Failed to build knowledge hierarchy: {e}")
            return {}
    
    def _calculate_hierarchy_levels(self, concept_network: Dict[str, Any]) -> None:
        """Calculate hierarchy levels for concepts."""
        # Find root concepts (no parents)
        root_concepts = [
            concept_id for concept_id, info in concept_network.items()
            if not info["parents"]
        ]
        
        # Calculate levels using BFS
        queue = [(concept_id, 0) for concept_id in root_concepts]
        visited = set()
        
        while queue:
            concept_id, level = queue.pop(0)
            if concept_id in visited:
                continue
            
            visited.add(concept_id)
            concept_network[concept_id]["level"] = level
            
            # Add children to queue
            for child_id in concept_network[concept_id]["children"]:
                if child_id not in visited:
                    queue.append((child_id, level + 1))
    
    def _generate_subject_description(
        self, 
        concepts: List[KeyConcept], 
        document_knowledge: List[DocumentKnowledge]
    ) -> str:
        """Generate a description of the subject."""
        try:
            # Create prompt for subject description
            concept_names = [c.name for c in concepts[:15]]  # Top 15 concepts
            doc_titles = [doc.title for doc in document_knowledge]
            
            prompt = f"""
            Generate a comprehensive description of the subject based on the following information:
            
            Subject Concepts: {', '.join(concept_names)}
            
            Source Documents: {', '.join(doc_titles)}
            
            Provide a 2-3 paragraph description that explains what this subject covers and its main themes.
            """
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate subject description: {e}")
            return "Subject description generation failed"
    
    def _determine_expertise_level(
        self, 
        concepts: List[KeyConcept], 
        relations: List[KnowledgeRelation]
    ) -> str:
        """Determine the expertise level required for this subject."""
        try:
            if not concepts:
                return "beginner"
            
            # Calculate complexity metrics
            avg_concept_confidence = sum(c.confidence for c in concepts) / len(concepts)
            num_relations = len(relations)
            concept_diversity = len(set(c.category for c in concepts))
            
            # Determine expertise level based on metrics
            if avg_concept_confidence > 0.8 and num_relations > 20 and concept_diversity > 5:
                return "advanced"
            elif avg_concept_confidence > 0.6 and num_relations > 10 and concept_diversity > 3:
                return "intermediate"
            else:
                return "beginner"
                
        except Exception as e:
            logger.warning(f"Failed to determine expertise level: {e}")
            return "intermediate"
    
    def _extract_domain_tags(
        self, 
        concepts: List[KeyConcept], 
        document_knowledge: List[DocumentKnowledge]
    ) -> List[str]:
        """Extract domain tags for the subject."""
        try:
            # Get concept categories
            categories = [c.category for c in concepts if c.category != "general"]
            
            # Get document themes
            all_themes = []
            for doc in document_knowledge:
                all_themes.extend(doc.main_themes)
            
            # Combine and deduplicate
            domain_tags = list(set(categories + all_themes))
            
            # Limit to top tags
            return domain_tags[:10]
            
        except Exception as e:
            logger.warning(f"Failed to extract domain tags: {e}")
            return []
    
    def _calculate_subject_quality_score(
        self, 
        concepts: List[KeyConcept], 
        relations: List[KnowledgeRelation]
    ) -> float:
        """Calculate quality score for the subject knowledge."""
        try:
            if not concepts:
                return 0.0
            
            # Concept quality (based on confidence and diversity)
            concept_scores = [c.confidence for c in concepts]
            avg_concept_confidence = sum(concept_scores) / len(concept_scores)
            
            # Relation quality (based on strength and coverage)
            if relations:
                relation_scores = [r.strength for r in relations]
                avg_relation_strength = sum(relation_scores) / len(relation_scores)
            else:
                avg_relation_strength = 0.0
            
            # Coverage score (how many concepts have relations)
            concepts_with_relations = set()
            for relation in relations:
                concepts_with_relations.add(relation.source_concept)
                concepts_with_relations.add(relation.target_concept)
            
            coverage_score = len(concepts_with_relations) / len(concepts) if concepts else 0.0
            
            # Diversity score (concept category diversity)
            categories = set(c.category for c in concepts)
            diversity_score = len(categories) / max(len(concepts), 1)
            
            # Overall quality score
            quality_score = (
                avg_concept_confidence * 0.3 + 
                avg_relation_strength * 0.25 + 
                coverage_score * 0.25 + 
                diversity_score * 0.2
            )
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate subject quality score: {e}")
            return 0.5
