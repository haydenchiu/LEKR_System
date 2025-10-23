"""
Tests for the logic_extractor extractor module.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from logic_extractor.extractor import (
    LogicExtractor, 
    add_logic_extraction_to_chunk, 
    add_logic_extraction_to_chunk_async,
    _initialize_llm
)
from logic_extractor.config import LogicExtractionConfig, DEFAULT_LOGIC_EXTRACTION_CONFIG
from logic_extractor.exceptions import (
    InvalidChunkError, 
    MissingAPIKeyError, 
    ChunkProcessingError,
    LLMInvocationError
)


class TestLogicExtractor:
    """Test cases for the LogicExtractor class."""
    
    def test_init_default(self, mock_llm):
        """Test LogicExtractor initialization with default config."""
        with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
            mock_chat_openai.return_value.with_structured_output.return_value = mock_llm
            
            extractor = LogicExtractor()
            
            assert extractor.config == DEFAULT_LOGIC_EXTRACTION_CONFIG
            assert extractor.llm == mock_llm
    
    def test_init_with_config(self, mock_llm):
        """Test LogicExtractor initialization with custom config."""
        config = LogicExtractionConfig(model_name="gpt-4", temperature=0.5)
        
        with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
            mock_chat_openai.return_value.with_structured_output.return_value = mock_llm
            
            extractor = LogicExtractor(config=config)
            
            assert extractor.config == config
            assert extractor.llm == mock_llm
    
    def test_init_with_llm(self, mock_llm):
        """Test LogicExtractor initialization with provided LLM."""
        extractor = LogicExtractor(llm=mock_llm)
        
        assert extractor.config == DEFAULT_LOGIC_EXTRACTION_CONFIG
        assert extractor.llm == mock_llm
    
    def test_init_llm_creation_error(self):
        """Test LogicExtractor initialization with LLM creation error."""
        with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
            mock_chat_openai.side_effect = Exception("API key missing")
            
            with pytest.raises(Exception):
                LogicExtractor()
    
    def test_extract_logic_success(self, sample_text_chunk, mock_llm, mock_logic_extraction_result):
        """Test successful logic extraction."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.invoke.return_value = mock_logic_extraction_result
        
        result = extractor.extract_logic(sample_text_chunk)
        
        assert hasattr(result, 'logic')
        assert result.logic == mock_logic_extraction_result
        mock_llm.invoke.assert_called_once()
    
    def test_extract_logic_table(self, sample_table_chunk, mock_llm, mock_logic_extraction_result):
        """Test logic extraction for table chunk."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.invoke.return_value = mock_logic_extraction_result
        
        result = extractor.extract_logic(sample_table_chunk)
        
        assert hasattr(result, 'logic')
        assert result.logic == mock_logic_extraction_result
        mock_llm.invoke.assert_called_once()
    
    def test_extract_logic_llm_error(self, sample_text_chunk, mock_llm):
        """Test logic extraction with LLM error."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(ChunkProcessingError):
            extractor.extract_logic(sample_text_chunk)
    
    def test_extract_logic_metadata_error(self, sample_text_chunk, mock_llm):
        """Test logic extraction with metadata error."""
        sample_text_chunk.metadata.to_dict.side_effect = Exception("Metadata error")
        extractor = LogicExtractor(llm=mock_llm)
        
        with pytest.raises(ChunkProcessingError):
            extractor.extract_logic(sample_text_chunk)
    
    @pytest.mark.asyncio
    async def test_extract_logic_async_success(self, sample_text_chunk, mock_llm, mock_logic_extraction_result):
        """Test successful async logic extraction."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.ainvoke.return_value = mock_logic_extraction_result
        
        result = await extractor.extract_logic_async(sample_text_chunk)
        
        assert hasattr(result, 'logic')
        assert result.logic == mock_logic_extraction_result
        mock_llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_logic_async_error(self, sample_text_chunk, mock_llm):
        """Test async logic extraction with error."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        
        with pytest.raises(ChunkProcessingError):
            await extractor.extract_logic_async(sample_text_chunk)
    
    def test_extract_logic_batch_success(self, sample_chunks, mock_llm, mock_logic_extraction_result):
        """Test successful batch logic extraction."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.invoke.return_value = mock_logic_extraction_result
        
        results = extractor.extract_logic_batch(sample_chunks)
        
        assert len(results) == len(sample_chunks)
        for result in results:
            assert hasattr(result, 'logic')
            assert result.logic == mock_logic_extraction_result
    
    def test_extract_logic_batch_with_errors(self, sample_chunks, mock_llm, mock_logic_extraction_result):
        """Test batch logic extraction with some chunks failing."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.invoke.side_effect = [Exception("Error 1"), mock_logic_extraction_result]
        
        results = extractor.extract_logic_batch(sample_chunks)
        
        assert len(results) == len(sample_chunks)
        # First chunk should be returned as-is due to error
        assert not hasattr(results[0], 'logic') or results[0].logic is None
        # Second chunk should have logic
        assert hasattr(results[1], 'logic')
    
    def test_extract_logic_batch_with_retries(self, sample_chunks, mock_llm, mock_logic_extraction_result):
        """Test batch logic extraction with retry configuration."""
        config = LogicExtractionConfig(max_retries=0)
        extractor = LogicExtractor(config=config, llm=mock_llm)
        mock_llm.invoke.return_value = mock_logic_extraction_result
        
        results = extractor.extract_logic_batch(sample_chunks)
        
        assert len(results) == len(sample_chunks)
    
    @pytest.mark.asyncio
    async def test_extract_logic_batch_async_success(self, sample_chunks, mock_llm, mock_logic_extraction_result):
        """Test successful async batch logic extraction."""
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.ainvoke.return_value = mock_logic_extraction_result
        
        results = await extractor.extract_logic_batch_async(sample_chunks)
        
        assert len(results) == len(sample_chunks)
        for result in results:
            assert hasattr(result, 'logic')
            assert result.logic == mock_logic_extraction_result
    
    @pytest.mark.asyncio
    async def test_extract_logic_batch_async_disabled(self, sample_chunks, mock_llm, mock_logic_extraction_result):
        """Test async batch logic extraction with async processing disabled."""
        # Since enable_async_processing is not in our config, we'll just test normal async behavior
        extractor = LogicExtractor(llm=mock_llm)
        mock_llm.ainvoke.return_value = mock_logic_extraction_result
        
        results = await extractor.extract_logic_batch_async(sample_chunks)
        
        assert len(results) == len(sample_chunks)
        mock_llm.ainvoke.assert_called()
    
    def test_get_extraction_stats(self, sample_chunks, mock_llm, mock_logic_extraction_result):
        """Test getting extraction statistics."""
        extractor = LogicExtractor(llm=mock_llm)
        
        # Add logic to some chunks
        sample_chunks[0].logic = mock_logic_extraction_result
        sample_chunks[1].logic = mock_logic_extraction_result
        
        stats = extractor.get_extraction_stats(sample_chunks)
        
        assert stats["total_chunks"] == 2
        assert stats["chunks_with_logic"] == 2
        assert stats["extraction_rate"] == 1.0
        assert stats["total_claims"] == 4  # 2 claims per extraction * 2 extractions
        assert stats["total_relations"] == 2  # 1 relation per extraction * 2 extractions
    
    def test_get_extraction_stats_empty(self, mock_llm):
        """Test getting extraction statistics for empty chunk list."""
        extractor = LogicExtractor(llm=mock_llm)
        stats = extractor.get_extraction_stats([])
        
        assert stats["total_chunks"] == 0
        assert stats["chunks_with_logic"] == 0
        assert stats["extraction_rate"] == 0.0
        assert stats["total_claims"] == 0
        assert stats["total_relations"] == 0
    
    def test_get_extraction_stats_all_enriched(self, sample_chunks, mock_logic_extraction_result, mock_llm):
        """Test getting extraction statistics when all chunks are enriched."""
        extractor = LogicExtractor(llm=mock_llm)
        
        # Add logic to all chunks
        for chunk in sample_chunks:
            chunk.logic = mock_logic_extraction_result
        
        stats = extractor.get_extraction_stats(sample_chunks)
        
        assert stats["total_chunks"] == 2
        assert stats["chunks_with_logic"] == 2
        assert stats["extraction_rate"] == 1.0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_add_logic_extraction_to_chunk_default(self, sample_text_chunk, mock_logic_extraction_result):
        """Test add_logic_extraction_to_chunk with default config."""
        with patch('logic_extractor.extractor._initialize_llm') as mock_init_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_logic_extraction_result
            mock_init_llm.return_value = mock_llm
            
            result = add_logic_extraction_to_chunk(sample_text_chunk)
            
            assert hasattr(result, 'logic')
            assert result.logic == mock_logic_extraction_result
            mock_llm.invoke.assert_called_once()
    
    def test_add_logic_extraction_to_chunk_custom_config(self, sample_text_chunk, mock_logic_extraction_result):
        """Test add_logic_extraction_to_chunk with custom config."""
        config = LogicExtractionConfig(model_name="gpt-4")
        
        with patch('logic_extractor.extractor._initialize_llm') as mock_init_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_logic_extraction_result
            mock_init_llm.return_value = mock_llm
            
            result = add_logic_extraction_to_chunk(sample_text_chunk, config)
            
            assert hasattr(result, 'logic')
            assert result.logic == mock_logic_extraction_result
            mock_init_llm.assert_called_once_with(config)
    
    @pytest.mark.asyncio
    async def test_add_logic_extraction_to_chunk_async_default(self, sample_text_chunk, mock_logic_extraction_result):
        """Test add_logic_extraction_to_chunk_async with default config."""
        with patch('logic_extractor.extractor._initialize_llm') as mock_init_llm:
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_logic_extraction_result)
            mock_init_llm.return_value = mock_llm
            
            result = await add_logic_extraction_to_chunk_async(sample_text_chunk)
            
            assert hasattr(result, 'logic')
            assert result.logic == mock_logic_extraction_result
            mock_llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_logic_extraction_to_chunk_async_custom_config(self, sample_text_chunk, mock_logic_extraction_result):
        """Test add_logic_extraction_to_chunk_async with custom config."""
        config = LogicExtractionConfig(model_name="gpt-4")
        
        with patch('logic_extractor.extractor._initialize_llm') as mock_init_llm:
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_logic_extraction_result)
            mock_init_llm.return_value = mock_llm
            
            result = await add_logic_extraction_to_chunk_async(sample_text_chunk, config)
            
            assert hasattr(result, 'logic')
            assert result.logic == mock_logic_extraction_result
            mock_init_llm.assert_called_once_with(config)


class TestExtractorIntegration:
    """Integration tests for the extractor module."""
    
    def test_full_logic_extraction_pipeline(self, sample_text_chunk, mock_logic_extraction_result):
        """Test full logic extraction pipeline."""
        with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_logic_extraction_result
            mock_chat_openai.return_value.with_structured_output.return_value = mock_llm
            
            extractor = LogicExtractor()
            result = extractor.extract_logic(sample_text_chunk)
            
            assert hasattr(result, 'logic')
            assert result.logic.chunk_id == "test_chunk_1"
            assert len(result.logic.claims) == 2
            assert len(result.logic.logical_relations) == 1
    
    @pytest.mark.asyncio
    async def test_async_full_logic_extraction_pipeline(self, sample_text_chunk, mock_logic_extraction_result):
        """Test full async logic extraction pipeline."""
        with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_logic_extraction_result)
            mock_chat_openai.return_value.with_structured_output.return_value = mock_llm
            
            extractor = LogicExtractor()
            result = await extractor.extract_logic_async(sample_text_chunk)
            
            assert hasattr(result, 'logic')
            assert result.logic.chunk_id == "test_chunk_1"
            assert len(result.logic.claims) == 2
            assert len(result.logic.logical_relations) == 1
    
    def test_logic_extraction_with_different_configs(self, sample_text_chunk, mock_logic_extraction_result):
        """Test logic extraction with different configurations."""
        configs = [
            LogicExtractionConfig(model_name="gpt-3.5-turbo", temperature=0.1),
            LogicExtractionConfig(model_name="gpt-4", temperature=0.0),
            LogicExtractionConfig(model_name="gpt-4o-mini", temperature=0.2)
        ]
        
        for config in configs:
            with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
                mock_llm = Mock()
                mock_llm.invoke.return_value = mock_logic_extraction_result
                mock_chat_openai.return_value.with_structured_output.return_value = mock_llm
                
                extractor = LogicExtractor(config=config)
                result = extractor.extract_logic(sample_text_chunk)
                
                assert hasattr(result, 'logic')
                assert result.logic == mock_logic_extraction_result
    
    def test_logic_extraction_error_handling(self, sample_text_chunk):
        """Test logic extraction error handling."""
        with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
            mock_chat_openai.side_effect = Exception("API key missing")
            
            with pytest.raises(Exception):
                LogicExtractor()
    
    def test_logic_extraction_statistics_accuracy(self, sample_chunks, mock_logic_extraction_result):
        """Test logic extraction statistics accuracy."""
        with patch('logic_extractor.extractor.ChatOpenAI') as mock_chat_openai:
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_logic_extraction_result
            mock_chat_openai.return_value.with_structured_output.return_value = mock_llm
            
            extractor = LogicExtractor()
            
            # Add logic to chunks manually
            sample_chunks[0].logic = mock_logic_extraction_result
            sample_chunks[1].logic = mock_logic_extraction_result
            
            stats = extractor.get_extraction_stats(sample_chunks)
            
            assert stats["total_chunks"] == 2
            assert stats["chunks_with_logic"] == 2
            assert stats["extraction_rate"] == 1.0
            assert stats["total_claims"] == 4  # 2 claims per extraction * 2 extractions
            assert stats["total_relations"] == 2  # 1 relation per extraction * 2 extractions
            assert stats["avg_claims_per_chunk"] == 2.0
            assert stats["avg_relations_per_chunk"] == 1.0
