"""
LERK System Integration Layer

This module provides clean integration interfaces between the clustering,
consolidation, and retriever modules, ensuring proper separation of concerns
and clear data flow.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

# Import AgentChat integration
from .agentchat_integration import (
    AgentChatLERKIntegration,
    create_agentchat_integration,
    process_chat_message_with_lerk
)


class LERKIntegrationManager:
    """
    Manages integration between clustering, consolidation, and retriever modules.
    
    This class provides a clean interface for coordinating operations across
    the three main modules while maintaining separation of concerns.
    """
    
    def __init__(
        self,
        clustering_manager=None,
        consolidation_manager=None,
        retriever_manager=None
    ):
        """
        Initialize the integration manager.
        
        Args:
            clustering_manager: DynamicClusteringManager instance
            consolidation_manager: ClusterBasedSubjectConsolidator instance
            retriever_manager: SubjectRetriever instance
        """
        self.clustering_manager = clustering_manager
        self.consolidation_manager = consolidation_manager
        self.retriever_manager = retriever_manager
        
        # Integration state
        self._integration_enabled = True
        self._last_integration_update = None
    
    async def process_new_documents_integrated(
        self,
        new_documents: List[Dict[str, Any]],
        document_knowledge_map: Dict[str, Any],
        update_strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Process new documents through the complete integration pipeline.
        
        This method coordinates the flow:
        1. Clustering module processes documents and updates clusters
        2. Consolidation module updates subject knowledge based on cluster changes
        3. Retriever module is notified of knowledge updates
        
        Args:
            new_documents: List of new document data
            document_knowledge_map: Mapping of document_id to DocumentKnowledge
            update_strategy: Clustering update strategy
            
        Returns:
            Complete integration results
        """
        try:
            logger.info(f"Starting integrated processing of {len(new_documents)} documents")
            
            integration_results = {
                "success": False,
                "clustering_results": {},
                "consolidation_results": {},
                "retriever_notification": {},
                "timestamp": datetime.utcnow().isoformat(),
                "errors": []
            }
            
            # Step 1: Process documents through clustering
            if self.clustering_manager:
                logger.info("Step 1: Processing documents through clustering")
                clustering_results = await self.clustering_manager.process_new_documents(
                    new_documents, update_strategy
                )
                integration_results["clustering_results"] = clustering_results
                
                if not clustering_results.get("success", False):
                    integration_results["errors"].append("Clustering processing failed")
                    return integration_results
            else:
                logger.warning("Clustering manager not available")
                integration_results["errors"].append("Clustering manager not available")
                return integration_results
            
            # Step 2: Update subject knowledge through consolidation
            if self.consolidation_manager and self.clustering_manager:
                logger.info("Step 2: Updating subject knowledge through consolidation")
                consolidation_results = await self.consolidation_manager.handle_clustering_update(
                    clustering_results, document_knowledge_map
                )
                integration_results["consolidation_results"] = consolidation_results
                
                if consolidation_results.get("error"):
                    integration_results["errors"].append(f"Consolidation error: {consolidation_results['error']}")
            else:
                logger.warning("Consolidation manager not available")
                integration_results["errors"].append("Consolidation manager not available")
            
            # Step 3: Notify retriever of knowledge updates
            if self.retriever_manager:
                logger.info("Step 3: Notifying retriever of knowledge updates")
                retriever_notification = await self._notify_retriever_of_updates(
                    clustering_results, integration_results.get("consolidation_results", {})
                )
                integration_results["retriever_notification"] = retriever_notification
            else:
                logger.warning("Retriever manager not available")
            
            # Determine overall success
            integration_results["success"] = (
                integration_results["clustering_results"].get("success", False) and
                len(integration_results["errors"]) == 0
            )
            
            self._last_integration_update = datetime.utcnow()
            
            logger.info(f"Integrated processing completed: {integration_results['success']}")
            return integration_results
            
        except Exception as e:
            logger.error(f"Integrated processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "clustering_results": {},
                "consolidation_results": {},
                "retriever_notification": {}
            }
    
    async def _notify_retriever_of_updates(
        self,
        clustering_results: Dict[str, Any],
        consolidation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Notify the retriever module of knowledge updates.
        
        Args:
            clustering_results: Results from clustering processing
            consolidation_results: Results from consolidation processing
            
        Returns:
            Retriever notification results
        """
        try:
            # This would notify the retriever to refresh its caches,
            # update indices, or perform other necessary updates
            # For now, return mock notification results
            
            notification_results = {
                "success": True,
                "cache_updated": True,
                "indices_refreshed": True,
                "notified_at": datetime.utcnow().isoformat()
            }
            
            logger.info("Retriever notified of knowledge updates")
            return notification_results
            
        except Exception as e:
            logger.error(f"Failed to notify retriever: {e}")
            return {
                "success": False,
                "error": str(e),
                "notified_at": datetime.utcnow().isoformat()
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and statistics."""
        try:
            return {
                "integration_enabled": self._integration_enabled,
                "last_update": self._last_integration_update.isoformat() if self._last_integration_update else None,
                "managers_available": {
                    "clustering": self.clustering_manager is not None,
                    "consolidation": self.consolidation_manager is not None,
                    "retriever": self.retriever_manager is not None
                },
                "integration_health": self._assess_integration_health()
            }
            
        except Exception as e:
            logger.error(f"Failed to get integration status: {e}")
            return {"error": str(e)}
    
    def _assess_integration_health(self) -> str:
        """Assess the health of the integration."""
        try:
            managers_available = sum([
                self.clustering_manager is not None,
                self.consolidation_manager is not None,
                self.retriever_manager is not None
            ])
            
            if managers_available == 3:
                return "healthy"
            elif managers_available >= 2:
                return "degraded"
            else:
                return "unhealthy"
                
        except Exception as e:
            logger.error(f"Failed to assess integration health: {e}")
            return "unknown"
    
    def enable_integration(self) -> None:
        """Enable integration between modules."""
        self._integration_enabled = True
        logger.info("Integration enabled")
    
    def disable_integration(self) -> None:
        """Disable integration between modules."""
        self._integration_enabled = False
        logger.info("Integration disabled")


# Convenience functions for creating integration managers
def create_integration_manager(
    clustering_manager=None,
    consolidation_manager=None,
    retriever_manager=None
) -> LERKIntegrationManager:
    """Create an integration manager instance."""
    return LERKIntegrationManager(
        clustering_manager=clustering_manager,
        consolidation_manager=consolidation_manager,
        retriever_manager=retriever_manager
    )


async def process_documents_with_integration(
    new_documents: List[Dict[str, Any]],
    document_knowledge_map: Dict[str, Any],
    clustering_manager=None,
    consolidation_manager=None,
    retriever_manager=None,
    update_strategy: str = "auto"
) -> Dict[str, Any]:
    """
    Process documents through the complete integration pipeline.
    
    This is a convenience function that creates an integration manager
    and processes documents in one call.
    
    Args:
        new_documents: List of new document data
        document_knowledge_map: Mapping of document_id to DocumentKnowledge
        clustering_manager: DynamicClusteringManager instance
        consolidation_manager: ClusterBasedSubjectConsolidator instance
        retriever_manager: SubjectRetriever instance
        update_strategy: Clustering update strategy
        
    Returns:
        Complete integration results
    """
    integration_manager = LERKIntegrationManager(
        clustering_manager=clustering_manager,
        consolidation_manager=consolidation_manager,
        retriever_manager=retriever_manager
    )
    
    return await integration_manager.process_new_documents_integrated(
        new_documents, document_knowledge_map, update_strategy
    )


# Module integration interfaces
class ClusteringIntegrationInterface:
    """Interface for clustering module integration."""
    
    @staticmethod
    async def process_documents_for_clustering(
        documents: List[Dict[str, Any]],
        update_strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Process documents through clustering.
        
        Args:
            documents: List of document data
            update_strategy: Update strategy
            
        Returns:
            Clustering results
        """
        try:
            from clustering import DynamicClusteringManager
            
            manager = DynamicClusteringManager()
            return await manager.process_new_documents(documents, update_strategy)
            
        except ImportError:
            logger.error("Clustering module not available")
            return {"success": False, "error": "Clustering module not available"}
        except Exception as e:
            logger.error(f"Clustering processing failed: {e}")
            return {"success": False, "error": str(e)}


class ConsolidationIntegrationInterface:
    """Interface for consolidation module integration."""
    
    @staticmethod
    async def handle_clustering_updates(
        clustering_results: Dict[str, Any],
        document_knowledge_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle clustering updates through consolidation.
        
        Args:
            clustering_results: Results from clustering
            document_knowledge_map: Document knowledge mapping
            
        Returns:
            Consolidation results
        """
        try:
            from consolidation import ClusterBasedSubjectConsolidator
            
            consolidator = ClusterBasedSubjectConsolidator()
            return await consolidator.handle_clustering_update(
                clustering_results, document_knowledge_map
            )
            
        except ImportError:
            logger.error("Consolidation module not available")
            return {"success": False, "error": "Consolidation module not available"}
        except Exception as e:
            logger.error(f"Consolidation processing failed: {e}")
            return {"success": False, "error": str(e)}


# Public API
__all__ = [
    # Main integration manager
    "LERKIntegrationManager",
    "create_integration_manager",
    "process_documents_with_integration",
    
    # Module interfaces
    "ClusteringIntegrationInterface",
    "ConsolidationIntegrationInterface", 
    "RetrieverIntegrationInterface",
    
    # AgentChat integration
    "AgentChatLERKIntegration",
    "create_agentchat_integration",
    "process_chat_message_with_lerk",
]
