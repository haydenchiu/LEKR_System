"""
Utility functions for knowledge consolidation.

This module provides utility functions for processing knowledge,
validating consistency, and performing various consolidation operations.
"""

import logging
from typing import List, Dict, Any, Set, Optional, Tuple
import re
from collections import defaultdict, Counter

from .models import DocumentKnowledge, SubjectKnowledge, KeyConcept, KnowledgeRelation
from .exceptions import KnowledgeValidationError

logger = logging.getLogger(__name__)


def extract_key_concepts(text: str, max_concepts: int = 10) -> List[str]:
    """
    Extract key concepts from text using simple heuristics.
    
    Args:
        text: Text to extract concepts from
        max_concepts: Maximum number of concepts to extract
        
    Returns:
        List of extracted concept names
    """
    try:
        # Simple concept extraction using capitalization and common patterns
        concepts = []
        
        # Find capitalized words and phrases
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend(capitalized_words)
        
        # Find quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', text)
        concepts.extend(quoted_terms)
        
        # Find terms in parentheses
        parenthetical_terms = re.findall(r'\(([^)]+)\)', text)
        concepts.extend(parenthetical_terms)
        
        # Remove duplicates and filter by length
        unique_concepts = list(set(concepts))
        filtered_concepts = [c for c in unique_concepts if len(c) > 2 and len(c) < 50]
        
        # Return top concepts by frequency
        concept_counts = Counter(filtered_concepts)
        top_concepts = [concept for concept, count in concept_counts.most_common(max_concepts)]
        
        return top_concepts
        
    except Exception as e:
        logger.warning(f"Failed to extract key concepts: {e}")
        return []


def build_knowledge_graph(
    concepts: List[KeyConcept], 
    relations: List[KnowledgeRelation]
) -> Dict[str, Any]:
    """
    Build a knowledge graph from concepts and relations.
    
    Args:
        concepts: List of key concepts
        relations: List of knowledge relations
        
    Returns:
        Dictionary representing the knowledge graph
    """
    try:
        # Create concept nodes
        nodes = {}
        for concept in concepts:
            nodes[concept.concept_id] = {
                "id": concept.concept_id,
                "name": concept.name,
                "description": concept.description,
                "category": concept.category,
                "confidence": concept.confidence,
                "keywords": concept.keywords
            }
        
        # Create relation edges
        edges = []
        for relation in relations:
            edge = {
                "id": relation.relation_id,
                "source": relation.source_concept,
                "target": relation.target_concept,
                "type": relation.relation_type,
                "strength": relation.strength,
                "description": relation.description,
                "evidence": relation.evidence
            }
            edges.append(edge)
        
        # Build adjacency lists
        adjacency = defaultdict(list)
        for relation in relations:
            adjacency[relation.source_concept].append({
                "target": relation.target_concept,
                "relation": relation
            })
            adjacency[relation.target_concept].append({
                "target": relation.source_concept,
                "relation": relation
            })
        
        # Calculate graph metrics
        num_nodes = len(nodes)
        num_edges = len(edges)
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Find connected components
        components = _find_connected_components(nodes.keys(), relations)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "adjacency": dict(adjacency),
            "metrics": {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": density,
                "num_components": len(components),
                "largest_component_size": max(len(comp) for comp in components) if components else 0
            },
            "components": components
        }
        
    except Exception as e:
        logger.warning(f"Failed to build knowledge graph: {e}")
        return {"nodes": {}, "edges": [], "adjacency": {}, "metrics": {}, "components": []}


def validate_knowledge_consistency(knowledge: Any) -> bool:
    """
    Validate the consistency of consolidated knowledge.
    
    Args:
        knowledge: DocumentKnowledge or SubjectKnowledge to validate
        
    Returns:
        True if knowledge is consistent, False otherwise
    """
    try:
        if isinstance(knowledge, DocumentKnowledge):
            return _validate_document_knowledge(knowledge)
        elif isinstance(knowledge, SubjectKnowledge):
            return _validate_subject_knowledge(knowledge)
        else:
            return False
            
    except Exception as e:
        logger.warning(f"Failed to validate knowledge consistency: {e}")
        return False


def merge_related_concepts(
    concepts: List[KeyConcept], 
    similarity_threshold: float = 0.8
) -> List[KeyConcept]:
    """
    Merge related concepts based on similarity.
    
    Args:
        concepts: List of concepts to merge
        similarity_threshold: Threshold for considering concepts similar
        
    Returns:
        List of merged concepts
    """
    try:
        if len(concepts) <= 1:
            return concepts
        
        # Simple similarity based on name and description overlap
        merged_concepts = []
        used_indices = set()
        
        for i, concept1 in enumerate(concepts):
            if i in used_indices:
                continue
            
            # Find similar concepts
            similar_concepts = [concept1]
            for j, concept2 in concepts[i+1:], i+1:
                if j in used_indices:
                    continue
                
                similarity = _calculate_concept_similarity(concept1, concept2)
                if similarity >= similarity_threshold:
                    similar_concepts.append(concept2)
                    used_indices.add(j)
            
            # Merge similar concepts
            if len(similar_concepts) > 1:
                merged_concept = _merge_concept_group(similar_concepts)
                merged_concepts.append(merged_concept)
            else:
                merged_concepts.append(concept1)
            
            used_indices.add(i)
        
        return merged_concepts
        
    except Exception as e:
        logger.warning(f"Failed to merge related concepts: {e}")
        return concepts


def calculate_knowledge_coverage(knowledge: Any) -> Dict[str, float]:
    """
    Calculate coverage metrics for knowledge.
    
    Args:
        knowledge: DocumentKnowledge or SubjectKnowledge
        
    Returns:
        Dictionary of coverage metrics
    """
    try:
        if isinstance(knowledge, DocumentKnowledge):
            concepts = knowledge.key_concepts
            relations = knowledge.knowledge_relations
        elif isinstance(knowledge, SubjectKnowledge):
            concepts = knowledge.core_concepts
            relations = knowledge.knowledge_relations
        else:
            return {}
        
        if not concepts:
            return {"concept_coverage": 0.0, "relation_coverage": 0.0, "overall_coverage": 0.0}
        
        # Concept coverage (based on confidence)
        concept_confidences = [c.confidence for c in concepts]
        concept_coverage = sum(concept_confidences) / len(concept_confidences)
        
        # Relation coverage (how many concepts have relations)
        concepts_with_relations = set()
        for relation in relations:
            concepts_with_relations.add(relation.source_concept)
            concepts_with_relations.add(relation.target_concept)
        
        relation_coverage = len(concepts_with_relations) / len(concepts)
        
        # Overall coverage
        overall_coverage = (concept_coverage + relation_coverage) / 2
        
        return {
            "concept_coverage": concept_coverage,
            "relation_coverage": relation_coverage,
            "overall_coverage": overall_coverage
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate knowledge coverage: {e}")
        return {"concept_coverage": 0.0, "relation_coverage": 0.0, "overall_coverage": 0.0}


def find_knowledge_gaps(knowledge: Any) -> List[str]:
    """
    Find potential gaps in knowledge.
    
    Args:
        knowledge: DocumentKnowledge or SubjectKnowledge
        
    Returns:
        List of identified gaps
    """
    try:
        gaps = []
        
        if isinstance(knowledge, DocumentKnowledge):
            concepts = knowledge.key_concepts
            relations = knowledge.knowledge_relations
        elif isinstance(knowledge, SubjectKnowledge):
            concepts = knowledge.core_concepts
            relations = knowledge.knowledge_relations
        else:
            return []
        
        # Check for isolated concepts (no relations)
        concept_ids = {c.concept_id for c in concepts}
        concepts_with_relations = set()
        for relation in relations:
            concepts_with_relations.add(relation.source_concept)
            concepts_with_relations.add(relation.target_concept)
        
        isolated_concepts = concept_ids - concepts_with_relations
        if isolated_concepts:
            gaps.append(f"Found {len(isolated_concepts)} isolated concepts with no relations")
        
        # Check for low-confidence concepts
        low_confidence_concepts = [c for c in concepts if c.confidence < 0.5]
        if low_confidence_concepts:
            gaps.append(f"Found {len(low_confidence_concepts)} low-confidence concepts")
        
        # Check for weak relations
        weak_relations = [r for r in relations if r.strength < 0.5]
        if weak_relations:
            gaps.append(f"Found {len(weak_relations)} weak relations")
        
        # Check for missing relation types
        relation_types = set(r.relation_type for r in relations)
        expected_types = {"causal", "hierarchical", "temporal", "associative"}
        missing_types = expected_types - relation_types
        if missing_types:
            gaps.append(f"Missing relation types: {', '.join(missing_types)}")
        
        return gaps
        
    except Exception as e:
        logger.warning(f"Failed to find knowledge gaps: {e}")
        return []


def _find_connected_components(concept_ids: List[str], relations: List[KnowledgeRelation]) -> List[List[str]]:
    """Find connected components in the knowledge graph."""
    try:
        # Build adjacency list
        adjacency = defaultdict(list)
        for relation in relations:
            adjacency[relation.source_concept].append(relation.target_concept)
            adjacency[relation.target_concept].append(relation.source_concept)
        
        # DFS to find components
        visited = set()
        components = []
        
        for concept_id in concept_ids:
            if concept_id not in visited:
                component = []
                stack = [concept_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(adjacency[current])
                
                if component:
                    components.append(component)
        
        return components
        
    except Exception as e:
        logger.warning(f"Failed to find connected components: {e}")
        return []


def _validate_document_knowledge(knowledge: DocumentKnowledge) -> bool:
    """Validate document knowledge consistency."""
    try:
        # Check required fields
        if not knowledge.document_id or not knowledge.title or not knowledge.summary:
            return False
        
        # Check concepts
        if not knowledge.key_concepts:
            return False
        
        # Check concept IDs are unique
        concept_ids = [c.concept_id for c in knowledge.key_concepts]
        if len(concept_ids) != len(set(concept_ids)):
            return False
        
        # Check relations reference valid concepts
        concept_id_set = set(concept_ids)
        for relation in knowledge.knowledge_relations:
            if (relation.source_concept not in concept_id_set or 
                relation.target_concept not in concept_id_set):
                return False
        
        # Check quality score is valid
        if not (0.0 <= knowledge.quality_score <= 1.0):
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to validate document knowledge: {e}")
        return False


def _validate_subject_knowledge(knowledge: SubjectKnowledge) -> bool:
    """Validate subject knowledge consistency."""
    try:
        # Check required fields
        if not knowledge.subject_id or not knowledge.name or not knowledge.description:
            return False
        
        # Check concepts
        if not knowledge.core_concepts:
            return False
        
        # Check concept IDs are unique
        concept_ids = [c.concept_id for c in knowledge.core_concepts]
        if len(concept_ids) != len(set(concept_ids)):
            return False
        
        # Check relations reference valid concepts
        concept_id_set = set(concept_ids)
        for relation in knowledge.knowledge_relations:
            if (relation.source_concept not in concept_id_set or 
                relation.target_concept not in concept_id_set):
                return False
        
        # Check quality score is valid
        if not (0.0 <= knowledge.quality_score <= 1.0):
            return False
        
        # Check expertise level is valid
        valid_levels = {"beginner", "intermediate", "advanced"}
        if knowledge.expertise_level not in valid_levels:
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to validate subject knowledge: {e}")
        return False


def _calculate_concept_similarity(concept1: KeyConcept, concept2: KeyConcept) -> float:
    """Calculate similarity between two concepts."""
    try:
        # Simple similarity based on name and description overlap
        name1_words = set(concept1.name.lower().split())
        name2_words = set(concept2.name.lower().split())
        
        desc1_words = set(concept1.description.lower().split())
        desc2_words = set(concept2.description.lower().split())
        
        # Calculate Jaccard similarity
        name_intersection = len(name1_words & name2_words)
        name_union = len(name1_words | name2_words)
        name_similarity = name_intersection / name_union if name_union > 0 else 0
        
        desc_intersection = len(desc1_words & desc2_words)
        desc_union = len(desc1_words | desc2_words)
        desc_similarity = desc_intersection / desc_union if desc_union > 0 else 0
        
        # Weighted average
        similarity = (name_similarity * 0.7 + desc_similarity * 0.3)
        
        return similarity
        
    except Exception as e:
        logger.warning(f"Failed to calculate concept similarity: {e}")
        return 0.0


def _merge_concept_group(concepts: List[KeyConcept]) -> KeyConcept:
    """Merge a group of similar concepts into one."""
    try:
        if len(concepts) == 1:
            return concepts[0]
        
        # Use the concept with highest confidence as base
        base_concept = max(concepts, key=lambda x: x.confidence)
        
        # Merge information
        merged_keywords = list(set([kw for c in concepts for kw in c.keywords]))
        merged_source_chunks = list(set([sc for c in concepts for sc in c.source_chunks]))
        
        # Merge descriptions (simple concatenation)
        merged_description = " | ".join([c.description for c in concepts])
        
        # Create merged concept
        merged_concept = KeyConcept(
            concept_id=base_concept.concept_id,
            name=base_concept.name,
            description=merged_description,
            category=base_concept.category,
            confidence=max(c.confidence for c in concepts),
            source_chunks=merged_source_chunks,
            keywords=merged_keywords,
            created_at=base_concept.created_at,
            updated_at=base_concept.updated_at
        )
        
        return merged_concept
        
    except Exception as e:
        logger.warning(f"Failed to merge concept group: {e}")
        return concepts[0]
