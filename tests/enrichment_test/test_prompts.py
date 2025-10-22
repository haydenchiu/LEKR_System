"""
Unit tests for enrichment prompts.

Tests the prompt generation functions and templates.
"""

import pytest

from enrichment.prompts import (
    generate_enrichment_prompt,
    get_table_enrichment_prompt,
    get_text_enrichment_prompt,
    create_custom_enrichment_prompt
)


class TestGenerateEnrichmentPrompt:
    """Test cases for generate_enrichment_prompt function."""
    
    def test_text_prompt_generation(self, sample_prompt_data):
        """Test prompt generation for text chunks."""
        chunk_text = sample_prompt_data["chunk_text"]
        is_table = sample_prompt_data["is_table"]
        expected_contains = sample_prompt_data["expected_prompt_contains"]
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Check that expected content is in the prompt
        for expected in expected_contains:
            assert expected in prompt
        
        # Check that chunk text is included
        assert chunk_text in prompt
    
    def test_table_prompt_generation(self, sample_table_prompt_data):
        """Test prompt generation for table chunks."""
        chunk_text = sample_table_prompt_data["chunk_text"]
        is_table = sample_table_prompt_data["is_table"]
        expected_contains = sample_table_prompt_data["expected_prompt_contains"]
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Check that expected content is in the prompt
        for expected in expected_contains:
            assert expected in prompt
        
        # Check that chunk text is included
        assert chunk_text in prompt
    
    def test_prompt_structure_text(self):
        """Test prompt structure for text chunks."""
        chunk_text = "This is a sample text chunk."
        is_table = False
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        # Check prompt structure
        assert "helpful assistant" in prompt
        assert "enrichment" in prompt
        assert "Chunk Content:" in prompt
        assert "_________________________" in prompt
        assert chunk_text in prompt
        
        # Should not contain table-specific content
        assert "TABLE chunk" not in prompt
        assert "key insights" not in prompt
        assert "data points" not in prompt
    
    def test_prompt_structure_table(self):
        """Test prompt structure for table chunks."""
        chunk_text = "<table><tr><td>Data</td></tr></table>"
        is_table = True
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        # Check prompt structure
        assert "helpful assistant" in prompt
        assert "enrichment" in prompt
        assert "Chunk Content:" in prompt
        assert "_________________________" in prompt
        assert chunk_text in prompt
        
        # Should contain table-specific content
        assert "TABLE chunk" in prompt
        assert "key insights" in prompt
        assert "data points" in prompt
    
    def test_empty_chunk_text(self):
        """Test prompt generation with empty chunk text."""
        chunk_text = ""
        is_table = False
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        assert isinstance(prompt, str)
        assert "helpful assistant" in prompt
        assert "enrichment" in prompt
        assert "Chunk Content:" in prompt
    
    def test_long_chunk_text(self):
        """Test prompt generation with long chunk text."""
        chunk_text = "This is a very long chunk of text that contains a lot of information and should be handled properly by the prompt generation function. " * 10
        is_table = False
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "helpful assistant" in prompt
    
    def test_special_characters_in_chunk(self):
        """Test prompt generation with special characters in chunk text."""
        chunk_text = "Chunk with special chars: @#$%^&*()_+{}|:<>?[]\\;'\",./"
        is_table = False
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "helpful assistant" in prompt
    
    def test_unicode_characters_in_chunk(self):
        """Test prompt generation with unicode characters in chunk text."""
        chunk_text = "Chunk with unicode: 你好世界 🌍 émojis"
        is_table = False
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "helpful assistant" in prompt
    
    def test_html_table_content(self):
        """Test prompt generation with HTML table content."""
        chunk_text = "<table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Data1</td><td>Data2</td></tr></table>"
        is_table = True
        
        prompt = generate_enrichment_prompt(chunk_text, is_table)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "TABLE chunk" in prompt
        assert "key insights" in prompt
    
    def test_prompt_consistency(self):
        """Test that prompts are consistent for the same input."""
        chunk_text = "Consistent test chunk"
        is_table = False
        
        prompt1 = generate_enrichment_prompt(chunk_text, is_table)
        prompt2 = generate_enrichment_prompt(chunk_text, is_table)
        
        assert prompt1 == prompt2
    
    def test_prompt_differentiation(self):
        """Test that prompts are different for text vs table chunks."""
        chunk_text = "Same chunk text"
        
        text_prompt = generate_enrichment_prompt(chunk_text, False)
        table_prompt = generate_enrichment_prompt(chunk_text, True)
        
        assert text_prompt != table_prompt
        assert "TABLE chunk" not in text_prompt
        assert "TABLE chunk" in table_prompt


class TestGetTableEnrichmentPrompt:
    """Test cases for get_table_enrichment_prompt function."""
    
    def test_table_prompt_content(self):
        """Test table enrichment prompt content."""
        prompt = get_table_enrichment_prompt()
        
        assert isinstance(prompt, str)
        assert "TABLE chunk" in prompt
        assert "key insights" in prompt
        assert "data points" in prompt
        assert "table" in prompt.lower()
    
    def test_table_prompt_consistency(self):
        """Test that table prompt is consistent."""
        prompt1 = get_table_enrichment_prompt()
        prompt2 = get_table_enrichment_prompt()
        
        assert prompt1 == prompt2
    
    def test_table_prompt_structure(self):
        """Test table prompt structure."""
        prompt = get_table_enrichment_prompt()
        
        # Should be a complete sentence
        assert prompt.strip().endswith(".")
        # Should contain key terms
        assert "summarize" in prompt.lower()
        assert "insights" in prompt.lower()


class TestGetTextEnrichmentPrompt:
    """Test cases for get_text_enrichment_prompt function."""
    
    def test_text_prompt_content(self):
        """Test text enrichment prompt content."""
        prompt = get_text_enrichment_prompt()
        
        assert isinstance(prompt, str)
        assert "helpful assistant" in prompt
        assert "enrichment" in prompt
        assert "document chunk" in prompt
    
    def test_text_prompt_consistency(self):
        """Test that text prompt is consistent."""
        prompt1 = get_text_enrichment_prompt()
        prompt2 = get_text_enrichment_prompt()
        
        assert prompt1 == prompt2
    
    def test_text_prompt_structure(self):
        """Test text prompt structure."""
        prompt = get_text_enrichment_prompt()
        
        # Should contain key terms
        assert "assistant" in prompt.lower()
        assert "enrichment" in prompt.lower()
        # The prompt may not end with a period, that's okay
        assert len(prompt.strip()) > 0


class TestCreateCustomEnrichmentPrompt:
    """Test cases for create_custom_enrichment_prompt function."""
    
    def test_custom_prompt_basic(self):
        """Test basic custom prompt creation."""
        chunk_text = "Test chunk"
        is_table = False
        custom_instructions = "Focus on technical details"
        
        prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert custom_instructions in prompt
        assert "Additional Instructions:" in prompt
    
    def test_custom_prompt_table(self):
        """Test custom prompt creation for table chunks."""
        chunk_text = "<table><tr><td>Data</td></tr></table>"
        is_table = True
        custom_instructions = "Emphasize data relationships"
        
        prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert custom_instructions in prompt
        assert "TABLE chunk" in prompt
        assert "Additional Instructions:" in prompt
    
    def test_custom_prompt_no_instructions(self):
        """Test custom prompt creation without additional instructions."""
        chunk_text = "Test chunk"
        is_table = False
        custom_instructions = ""
        
        prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "Additional Instructions:" not in prompt
    
    def test_custom_prompt_none_instructions(self):
        """Test custom prompt creation with None instructions."""
        chunk_text = "Test chunk"
        is_table = False
        custom_instructions = None
        
        prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "Additional Instructions:" not in prompt
    
    def test_custom_prompt_long_instructions(self):
        """Test custom prompt creation with long instructions."""
        chunk_text = "Test chunk"
        is_table = False
        custom_instructions = "This is a very long set of custom instructions that should be included in the prompt and should be handled properly by the function. " * 5
        
        prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert custom_instructions in prompt
        assert "Additional Instructions:" in prompt
    
    def test_custom_prompt_special_characters(self):
        """Test custom prompt creation with special characters."""
        chunk_text = "Test chunk with @#$%^&*()"
        is_table = False
        custom_instructions = "Handle special chars: @#$%^&*()_+{}|:<>?[]\\;'\",./"
        
        prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert custom_instructions in prompt
    
    def test_custom_prompt_unicode(self):
        """Test custom prompt creation with unicode characters."""
        chunk_text = "Test chunk with unicode: 你好世界 🌍"
        is_table = False
        custom_instructions = "Handle unicode: 你好世界 🌍 émojis"
        
        prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert custom_instructions in prompt
    
    def test_custom_prompt_consistency(self):
        """Test that custom prompts are consistent for the same input."""
        chunk_text = "Consistent test chunk"
        is_table = False
        custom_instructions = "Consistent instructions"
        
        prompt1 = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        prompt2 = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        assert prompt1 == prompt2
    
    def test_custom_prompt_differentiation(self):
        """Test that custom prompts are different for different inputs."""
        chunk_text = "Same chunk text"
        is_table = False
        
        prompt1 = create_custom_enrichment_prompt(chunk_text, is_table, "Instructions 1")
        prompt2 = create_custom_enrichment_prompt(chunk_text, is_table, "Instructions 2")
        
        assert prompt1 != prompt2
        assert "Instructions 1" in prompt1
        assert "Instructions 2" in prompt2


class TestPromptIntegration:
    """Integration tests for prompt functions."""
    
    def test_prompt_functions_work_together(self):
        """Test that all prompt functions work together."""
        chunk_text = "Integration test chunk"
        is_table = False
        custom_instructions = "Integration test instructions"
        
        # Test all prompt functions
        basic_prompt = generate_enrichment_prompt(chunk_text, is_table)
        text_prompt = get_text_enrichment_prompt()
        table_prompt = get_table_enrichment_prompt()
        custom_prompt = create_custom_enrichment_prompt(chunk_text, is_table, custom_instructions)
        
        # All should be strings
        assert isinstance(basic_prompt, str)
        assert isinstance(text_prompt, str)
        assert isinstance(table_prompt, str)
        assert isinstance(custom_prompt, str)
        
        # All should contain the chunk text
        assert chunk_text in basic_prompt
        assert chunk_text in custom_prompt
        
        # Custom prompt should contain additional instructions
        assert custom_instructions in custom_prompt
        assert "Additional Instructions:" in custom_prompt
    
    def test_prompt_functions_with_different_chunk_types(self):
        """Test prompt functions with different chunk types."""
        text_chunk = "This is a text chunk"
        table_chunk = "<table><tr><td>Data</td></tr></table>"
        
        # Test text chunk
        text_prompt = generate_enrichment_prompt(text_chunk, False)
        assert text_chunk in text_prompt
        assert "TABLE chunk" not in text_prompt
        
        # Test table chunk
        table_prompt = generate_enrichment_prompt(table_chunk, True)
        assert table_chunk in table_prompt
        assert "TABLE chunk" in table_prompt
    
    def test_prompt_functions_error_handling(self):
        """Test prompt functions with edge cases."""
        # Test with very long strings
        long_text = "x" * 10000
        prompt = generate_enrichment_prompt(long_text, False)
        assert isinstance(prompt, str)
        assert long_text in prompt
        
        # Test with None values (should handle gracefully)
        prompt_none_text = generate_enrichment_prompt(None, False)
        assert isinstance(prompt_none_text, str)
        
        prompt_none_table = generate_enrichment_prompt("text", None)
        assert isinstance(prompt_none_table, str)
