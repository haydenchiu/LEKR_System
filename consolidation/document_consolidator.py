"""
Document-level knowledge consolidation.

This module implements document-level consolidation functionality that
summarizes logic chunks into key concepts and builds document knowledge.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import Runnable
    from langchain_core.prompts import ChatPromptTemplate
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError as e:
    raise ImportError(
        "Required consolidation dependencies not installed. "
        "Please install: pip install langchain-openai sentence-transformers scikit-learn"
    ) from e

from .models import DocumentKnowledge, KeyConcept, KnowledgeRelation
from .config import ConsolidationConfig, DEFAULT_CONSOLIDATION_CONFIG
from .exceptions import (
    DocumentConsolidationError, ConceptExtractionError, RelationExtractionError,
    MissingAPIKeyError, KnowledgeValidationError
)
from .utils import extract_key_concepts, build_knowledge_graph, validate_knowledge_consistency

logger = logging.getLogger(__name__)


class DocumentConsolidator:
    """
    Consolidates knowledge from document chunks into document-level knowledge.
    
    This class handles the process of extracting key concepts and relations
    from document chunks and building a comprehensive document knowledge representation.
    """
    
    def __init__(self, config: ConsolidationConfig = DEFAULT_CONSOLIDATION_CONFIG):
        """
        Initialize the DocumentConsolidator.
        
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
            
            logger.info(f"Initialized DocumentConsolidator with model: {self.config.model_name}")
            
        except Exception as e:
            raise MissingAPIKeyError(f"Failed to initialize models: {e}") from e
    
    def consolidate_document(
        self, 
        document_id: str,
        document_title: str,
        chunks: List[Any],
        chunk_logic_data: Optional[List[Any]] = None
    ) -> DocumentKnowledge:
        """
        Consolidate knowledge from document chunks.
        
        Args:
            document_id: Unique identifier for the document
            document_title: Title of the document
            chunks: List of document chunks
            chunk_logic_data: Optional logic extraction data for chunks
            
        Returns:
            DocumentKnowledge containing consolidated knowledge
            
        Raises:
            DocumentConsolidationError: If consolidation fails
        """
        try:
            logger.info(f"Starting document consolidation for document: {document_id}")
            
            # Extract key concepts from chunks
            key_concepts = self._extract_concepts_from_chunks(chunks, chunk_logic_data)
            
            # Extract relations between concepts
            knowledge_relations = self._extract_concept_relations(key_concepts, chunks)
            
            # Build knowledge graph
            knowledge_graph = build_knowledge_graph(key_concepts, knowledge_relations)
            
            # Generate document summary
            summary = self._generate_document_summary(chunks, key_concepts)
            
            # Extract main themes
            main_themes = self._extract_main_themes(key_concepts)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(key_concepts, knowledge_relations)
            
            # Create document knowledge
            document_knowledge = DocumentKnowledge(
                document_id=document_id,
                title=document_title,
                summary=summary,
                key_concepts=key_concepts,
                knowledge_relations=knowledge_relations,
                main_themes=main_themes,
                knowledge_graph=knowledge_graph,
                quality_score=quality_score,
                consolidation_metadata={
                    "num_chunks": len(chunks),
                    "num_concepts": len(key_concepts),
                    "num_relations": len(knowledge_relations),
                    "consolidation_timestamp": datetime.now().isoformat(),
                    "config_used": self.config.model_dump()
                }
            )
            
            # Validate knowledge if enabled
            if self.config.enable_quality_validation:
                if not validate_knowledge_consistency(document_knowledge):
                    raise KnowledgeValidationError("Knowledge validation failed")
            
            logger.info(f"Successfully consolidated document {document_id} with {len(key_concepts)} concepts")
            return document_knowledge
            
        except Exception as e:
            raise DocumentConsolidationError(f"Failed to consolidate document {document_id}: {e}") from e
    
    def _extract_concepts_from_chunks(
        self, 
        chunks: List[Any], 
        chunk_logic_data: Optional[List[Any]] = None
    ) -> List[KeyConcept]:
        """Extract key concepts from document chunks."""
        try:
            concepts = []
            concept_counter = 0
            
            for i, chunk in enumerate(chunks):
                # Get chunk content
                chunk_content = self._get_chunk_content(chunk)
                logic_data = chunk_logic_data[i] if chunk_logic_data and i < len(chunk_logic_data) else None
                
                # Extract concepts from chunk
                chunk_concepts = self._extract_concepts_from_chunk(chunk_content, logic_data, i)
                
                # Add source chunk information
                for concept in chunk_concepts:
                    concept.concept_id = f"doc_{concept_counter}"
                    concept.source_chunks.append(f"chunk_{i}")
                    concepts.append(concept)
                    concept_counter += 1
            
            # Merge similar concepts
            merged_concepts = self._merge_similar_concepts(concepts)
            
            # Filter by confidence threshold
            filtered_concepts = [
                concept for concept in merged_concepts 
                if concept.confidence >= self.config.min_concept_confidence
            ]
            
            # Limit number of concepts
            if len(filtered_concepts) > self.config.max_concepts_per_document:
                filtered_concepts = sorted(
                    filtered_concepts, 
                    key=lambda x: x.confidence, 
                    reverse=True
                )[:self.config.max_concepts_per_document]
            
            return filtered_concepts
            
        except Exception as e:
            raise ConceptExtractionError(f"Failed to extract concepts: {e}") from e
    
    def _extract_concepts_from_chunk(
        self, 
        chunk_content: str, 
        logic_data: Optional[Any], 
        chunk_index: int
    ) -> List[KeyConcept]:
        """Extract concepts from a single chunk."""
        try:
            # Create prompt for concept extraction
            prompt = self._create_concept_extraction_prompt(chunk_content, logic_data)
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            concepts_text = response.content
            
            # Parse concepts from response
            concepts = self._parse_concepts_from_response(concepts_text, chunk_index)
            
            return concepts
            
        except Exception as e:
            logger.warning(f"Failed to extract concepts from chunk {chunk_index}: {e}")
            return []
    
    def _create_concept_extraction_prompt(self, chunk_content: str, logic_data: Optional[Any]) -> str:
        """Create prompt for concept extraction."""
        logic_context = ""
        if logic_data:
            logic_context = f"\n\nLogic Data:\n{logic_data}"
        
        prompt = f"""
        Extract key concepts from the following document chunk. For each concept, provide:
        1. Name/title
        2. Description
        3. Category/domain
        4. Confidence score (0.0-1.0)
        5. Keywords
        
        Chunk Content:
        {chunk_content}
        {logic_context}
        
        Format your response as a JSON list of concepts:
        [
            {{
                "name": "Concept Name",
                "description": "Detailed description",
                "category": "Domain/Category",
                "confidence": 0.8,
                "keywords": ["keyword1", "keyword2"]
            }}
        ]
        """
        return prompt
    
    def _parse_concepts_from_response(self, response_text: str, chunk_index: int) -> List[KeyConcept]:
        """Parse concepts from LLM response."""
        try:
            import json
            concepts_data = json.loads(response_text)
            concepts = []
            
            for concept_data in concepts_data:
                concept = KeyConcept(
                    concept_id=f"temp_{chunk_index}_{len(concepts)}",
                    name=concept_data.get("name", ""),
                    description=concept_data.get("description", ""),
                    category=concept_data.get("category", "general"),
                    confidence=float(concept_data.get("confidence", 0.5)),
                    source_chunks=[],
                    keywords=concept_data.get("keywords", [])
                )
                concepts.append(concept)
            
            return concepts
            
        except Exception as e:
            logger.warning(f"Failed to parse concepts from response: {e}")
            return []
    
    def _extract_concept_relations(
        self, 
        concepts: List[KeyConcept], 
        chunks: List[Any]
    ) -> List[KnowledgeRelation]:
        """Extract relations between concepts."""
        try:
            relations = []
            relation_counter = 0
            
            # Extract relations from chunks
            for i, chunk in enumerate(chunks):
                chunk_content = self._get_chunk_content(chunk)
                chunk_relations = self._extract_relations_from_chunk(
                    chunk_content, concepts, i
                )
                relations.extend(chunk_relations)
                relation_counter += len(chunk_relations)
            
            # Filter and merge relations
            filtered_relations = self._filter_and_merge_relations(relations)
            
            # Limit number of relations
            if len(filtered_relations) > self.config.max_relations_per_document:
                filtered_relations = sorted(
                    filtered_relations,
                    key=lambda x: x.strength,
                    reverse=True
                )[:self.config.max_relations_per_document]
            
            return filtered_relations
            
        except Exception as e:
            raise RelationExtractionError(f"Failed to extract relations: {e}") from e
    
    def _extract_relations_from_chunk(
        self, 
        chunk_content: str, 
        concepts: List[KeyConcept], 
        chunk_index: int
    ) -> List[KnowledgeRelation]:
        """Extract relations from a single chunk."""
        try:
            # Create prompt for relation extraction
            prompt = self._create_relation_extraction_prompt(chunk_content, concepts)
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            relations_text = response.content
            
            # Parse relations from response
            relations = self._parse_relations_from_response(relations_text, concepts)
            
            return relations
            
        except Exception as e:
            logger.warning(f"Failed to extract relations from chunk {chunk_index}: {e}")
            return []
    
    def _create_relation_extraction_prompt(self, chunk_content: str, concepts: List[KeyConcept]) -> str:
        """Create prompt for relation extraction."""
        concept_list = "\n".join([f"- {c.name} (ID: {c.concept_id})" for c in concepts])
        
        prompt = f"""
        Identify relationships between the following concepts in the document chunk.
        For each relationship, provide:
        1. Source concept ID
        2. Target concept ID
        3. Relationship type (causal, hierarchical, temporal, associative, etc.)
        4. Strength (0.0-1.0)
        5. Description
        6. Evidence from the text
        
        Available Concepts:
        {concept_list}
        
        Chunk Content:
        {chunk_content}
        
        Format your response as a JSON list of relations:
        [
            {{
                "source_concept": "concept_id_1",
                "target_concept": "concept_id_2",
                "relation_type": "causal",
                "strength": 0.8,
                "description": "Description of the relationship",
                "evidence": ["evidence1", "evidence2"]
            }}
        ]
        """
        return prompt
    
    def _parse_relations_from_response(
        self, 
        response_text: str, 
        concepts: List[KeyConcept]
    ) -> List[KnowledgeRelation]:
        """Parse relations from LLM response."""
        try:
            import json
            relations_data = json.loads(response_text)
            relations = []
            
            # Create concept ID mapping
            concept_ids = {c.concept_id for c in concepts}
            
            for relation_data in relations_data:
                source_id = relation_data.get("source_concept")
                target_id = relation_data.get("target_concept")
                
                # Validate concept IDs
                if source_id in concept_ids and target_id in concept_ids:
                    relation = KnowledgeRelation(
                        relation_id=f"rel_{len(relations)}",
                        source_concept=source_id,
                        target_concept=target_id,
                        relation_type=relation_data.get("relation_type", "associative"),
                        strength=float(relation_data.get("strength", 0.5)),
                        description=relation_data.get("description", ""),
                        evidence=relation_data.get("evidence", [])
                    )
                    relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.warning(f"Failed to parse relations from response: {e}")
            return []
    
    def _merge_similar_concepts(self, concepts: List[KeyConcept]) -> List[KeyConcept]:
        """Merge similar concepts based on similarity threshold."""
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
                    merged_concept = self._merge_concept_group([concept] + [concepts[j] for j in similar_indices])
                    merged_concepts.append(merged_concept)
                    merged_indices.update([i] + similar_indices)
                else:
                    merged_concepts.append(concept)
            
            return merged_concepts
            
        except Exception as e:
            logger.warning(f"Failed to merge similar concepts: {e}")
            return concepts
    
    def _merge_concept_group(self, concepts: List[KeyConcept]) -> KeyConcept:
        """Merge a group of similar concepts into one."""
        if len(concepts) == 1:
            return concepts[0]
        
        # Use the concept with highest confidence as base
        base_concept = max(concepts, key=lambda x: x.confidence)
        
        # Merge information
        merged_keywords = list(set([kw for c in concepts for kw in c.keywords]))
        merged_source_chunks = list(set([sc for c in concepts for sc in c.source_chunks]))
        
        # Create merged concept
        merged_concept = KeyConcept(
            concept_id=base_concept.concept_id,
            name=base_concept.name,
            description=base_concept.description,
            category=base_concept.category,
            confidence=max(c.confidence for c in concepts),
            source_chunks=merged_source_chunks,
            keywords=merged_keywords,
            created_at=base_concept.created_at,
            updated_at=datetime.now()
        )
        
        return merged_concept
    
    def _filter_and_merge_relations(self, relations: List[KnowledgeRelation]) -> List[KnowledgeRelation]:
        """Filter and merge duplicate relations."""
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
                merged_relation = self._merge_relation_group(group)
                merged_relations.append(merged_relation)
        
        # Filter by strength threshold
        filtered_relations = [
            rel for rel in merged_relations 
            if rel.strength >= self.config.relation_strength_threshold
        ]
        
        return filtered_relations
    
    def _merge_relation_group(self, relations: List[KnowledgeRelation]) -> KnowledgeRelation:
        """Merge a group of relations into one."""
        if len(relations) == 1:
            return relations[0]
        
        # Use the relation with highest strength as base
        base_relation = max(relations, key=lambda x: x.strength)
        
        # Merge information
        merged_evidence = list(set([ev for r in relations for ev in r.evidence]))
        
        # Create merged relation
        merged_relation = KnowledgeRelation(
            relation_id=base_relation.relation_id,
            source_concept=base_relation.source_concept,
            target_concept=base_relation.target_concept,
            relation_type=base_relation.relation_type,
            strength=max(r.strength for r in relations),
            description=base_relation.description,
            evidence=merged_evidence,
            created_at=base_relation.created_at
        )
        
        return merged_relation
    
    def _generate_document_summary(self, chunks: List[Any], concepts: List[KeyConcept]) -> str:
        """Generate a summary of the document."""
        try:
            # Create prompt for document summary
            concept_names = [c.name for c in concepts[:10]]  # Top 10 concepts
            chunk_summaries = [self._get_chunk_content(chunk)[:200] for chunk in chunks[:5]]  # First 5 chunks
            
            prompt = f"""
            Generate a comprehensive summary of the document based on the following information:
            
            Key Concepts: {', '.join(concept_names)}
            
            Document Chunks:
            {chr(10).join(chunk_summaries)}
            
            Provide a 2-3 paragraph summary that captures the main themes and key insights.
            """
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate document summary: {e}")
            return "Summary generation failed"
    
    def _extract_main_themes(self, concepts: List[KeyConcept]) -> List[str]:
        """Extract main themes from concepts."""
        try:
            # Group concepts by category
            category_groups = {}
            for concept in concepts:
                category = concept.category
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(concept)
            
            # Get top categories by concept count and confidence
            theme_scores = {}
            for category, cat_concepts in category_groups.items():
                score = len(cat_concepts) * sum(c.confidence for c in cat_concepts) / len(cat_concepts)
                theme_scores[category] = score
            
            # Return top themes
            sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
            return [theme for theme, score in sorted_themes[:5]]
            
        except Exception as e:
            logger.warning(f"Failed to extract main themes: {e}")
            return []
    
    def _calculate_quality_score(
        self, 
        concepts: List[KeyConcept], 
        relations: List[KnowledgeRelation]
    ) -> float:
        """Calculate quality score for the consolidated knowledge."""
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
            
            # Overall quality score
            quality_score = (avg_concept_confidence * 0.4 + avg_relation_strength * 0.3 + coverage_score * 0.3)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate quality score: {e}")
            return 0.5
    
    def _get_chunk_content(self, chunk: Any) -> str:
        """Extract text content from a chunk."""
        try:
            if hasattr(chunk, 'text'):
                return chunk.text
            elif hasattr(chunk, 'content'):
                return chunk.content
            else:
                return str(chunk)
        except Exception:
            return str(chunk)
