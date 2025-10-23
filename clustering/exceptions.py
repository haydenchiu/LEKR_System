"""
Custom exception classes for the clustering module.

This module defines a hierarchy of custom exceptions to provide more specific
error handling for various scenarios that can occur during clustering,
such as issues with document processing, clustering initialization, or cluster assignment.
"""


class ClusteringError(Exception):
    """Base exception for clustering module errors."""
    pass


class InvalidDocumentError(ClusteringError):
    """Raised when an invalid document is provided for clustering."""
    pass


class ClusteringInitializationError(ClusteringError):
    """Raised when clustering model initialization fails."""
    pass


class ClusterAssignmentError(ClusteringError):
    """Raised when there is an error assigning documents to clusters."""
    pass


class ClusteringFitError(ClusteringError):
    """Raised when clustering fitting fails."""
    pass


class ClusterNotFoundError(ClusteringError):
    """Raised when a requested cluster is not found."""
    pass


class DocumentNotFoundError(ClusteringError):
    """Raised when a requested document is not found in clustering results."""
    pass


class ClusteringQualityError(ClusteringError):
    """Raised when clustering quality is below acceptable thresholds."""
    pass
