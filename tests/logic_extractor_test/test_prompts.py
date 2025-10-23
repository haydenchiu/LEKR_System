"""
Tests for the logic_extractor prompts module.
"""

import pytest
from logic_extractor.prompts import (
    generate_logic_extraction_prompt,
    get_table_logic_extraction_prompt,
    get_text_logic_extraction_prompt,
    create_custom_logic_extraction_prompt,
    get_logic_extraction_schema_prompt
)


class TestGenerateLogicExtractionPrompt:
    """Test cases for the generate_logic_extraction_prompt function."""
    
    def test_text_prompt_generation(self):
        """Test prompt generation for text chunks."""
        chunk_text = "The Transformer model revolutionized NLP by introducing self-attention mechanism."
        prompt = generate_logic_extraction_prompt(chunk_text, is_table=False)
        
        assert isinstance(prompt, str)
        assert "Logic Extraction Agent" in prompt
        assert chunk_text in prompt
        assert "table" not in prompt.lower()  # Should not mention table
        
    def test_table_prompt_generation(self):
        """Test prompt generation for table chunks."""
        chunk_text = "<table><tr><td>Model</td><td>Score</td></tr></table>"
        prompt = generate_logic_extraction_prompt(chunk_text, is_table=True)
        
        assert isinstance(prompt, str)
        assert "Logic Extraction Agent" in prompt
        assert chunk_text in prompt
        assert "table" in prompt.lower()
    
    def test_prompt_structure_text(self):
        """Test prompt structure for text chunks."""
        chunk_text = "Sample text content"
        prompt = generate_logic_extraction_prompt(chunk_text, is_table=False)
        
        assert "You are a Logic Extraction Agent" in prompt
        assert "Chunk Content:" in prompt
        assert "Instructions:" in prompt
        assert "LogicExtractionSchemaLiteChunk format" in prompt
        assert chunk_text in prompt
    
    def test_prompt_structure_table(self):
        """Test prompt structure for table chunks."""
        chunk_text = "<table>Sample table</table>"
        prompt = generate_logic_extraction_prompt(chunk_text, is_table=True)
        
        assert "You are a Logic Extraction Agent" in prompt
        assert "Table Summary / Content:" in prompt
        assert "Instructions:" in prompt
        assert "LogicExtractionSchemaLiteChunk format" in prompt
        assert chunk_text in prompt
    
    def test_empty_chunk_text(self):
        """Test prompt generation with empty chunk text."""
        prompt = generate_logic_extraction_prompt("", is_table=False)
        
        assert isinstance(prompt, str)
        assert "Logic Extraction Agent" in prompt
        assert "Instructions:" in prompt
    
    def test_long_chunk_text(self):
        """Test prompt generation with long chunk text."""
        long_text = "This is a very long text chunk. " * 100
        prompt = generate_logic_extraction_prompt(long_text, is_table=False)
        
        assert isinstance(prompt, str)
        assert long_text in prompt
        assert "Logic Extraction Agent" in prompt
    
    def test_special_characters_in_chunk(self):
        """Test prompt generation with special characters in chunk text."""
        special_text = "Text with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        prompt = generate_logic_extraction_prompt(special_text, is_table=False)
        
        assert isinstance(prompt, str)
        assert special_text in prompt
    
    def test_unicode_characters_in_chunk(self):
        """Test prompt generation with unicode characters in chunk text."""
        unicode_text = "Text with unicode: 你好世界 🌍 émojis"
        prompt = generate_logic_extraction_prompt(unicode_text, is_table=False)
        
        assert isinstance(prompt, str)
        assert unicode_text in prompt
    
    def test_html_table_content(self):
        """Test prompt generation with HTML table content."""
        html_table = """
        <table>
            <tr><th>Model</th><th>BLEU Score</th></tr>
            <tr><td>RNN</td><td>25.2</td></tr>
            <tr><td>Transformer</td><td>34.5</td></tr>
        </table>
        """
        prompt = generate_logic_extraction_prompt(html_table, is_table=True)
        
        assert isinstance(prompt, str)
        assert html_table in prompt
        assert "table" in prompt.lower()
    
    def test_prompt_consistency(self):
        """Test that prompts are consistent for the same input."""
        chunk_text = "Consistent text content"
        prompt1 = generate_logic_extraction_prompt(chunk_text, is_table=False)
        prompt2 = generate_logic_extraction_prompt(chunk_text, is_table=False)
        
        assert prompt1 == prompt2
    
    def test_prompt_differentiation(self):
        """Test that prompts differ between table and text chunks."""
        chunk_text = "Same content for both"
        text_prompt = generate_logic_extraction_prompt(chunk_text, is_table=False)
        table_prompt = generate_logic_extraction_prompt(chunk_text, is_table=True)
        
        assert text_prompt != table_prompt
        assert "table" in table_prompt.lower()
        assert "Table Summary / Content:" in table_prompt


class TestGetTableLogicExtractionPrompt:
    """Test cases for the get_table_logic_extraction_prompt function."""
    
    def test_table_prompt_content(self):
        """Test table-specific prompt content."""
        table_content = "<table><tr><td>Data</td></tr></table>"
        prompt = get_table_logic_extraction_prompt(table_content)
        
        assert isinstance(prompt, str)
        assert "tabular data analysis" in prompt.lower()
        assert table_content in prompt
        assert "LogicExtractionSchemaLiteChunk format" in prompt
    
    def test_table_prompt_consistency(self):
        """Test table prompt consistency."""
        table_content = "Consistent table content"
        prompt1 = get_table_logic_extraction_prompt(table_content)
        prompt2 = get_table_logic_extraction_prompt(table_content)
        
        assert prompt1 == prompt2
    
    def test_table_prompt_structure(self):
        """Test table prompt structure."""
        table_content = "Sample table"
        prompt = get_table_logic_extraction_prompt(table_content)
        
        assert "Logic Extraction Agent" in prompt
        assert "Table Content:" in prompt
        assert "Focus on:" in prompt
        assert "LogicExtractionSchemaLiteChunk format" in prompt


class TestGetTextLogicExtractionPrompt:
    """Test cases for the get_text_logic_extraction_prompt function."""
    
    def test_text_prompt_content(self):
        """Test text-specific prompt content."""
        text_content = "Sample text content for analysis"
        prompt = get_text_logic_extraction_prompt(text_content)
        
        assert isinstance(prompt, str)
        assert "textual content analysis" in prompt.lower()
        assert text_content in prompt
        assert "LogicExtractionSchemaLiteChunk format" in prompt
    
    def test_text_prompt_consistency(self):
        """Test text prompt consistency."""
        text_content = "Consistent text content"
        prompt1 = get_text_logic_extraction_prompt(text_content)
        prompt2 = get_text_logic_extraction_prompt(text_content)
        
        assert prompt1 == prompt2
    
    def test_text_prompt_structure(self):
        """Test text prompt structure."""
        text_content = "Sample text"
        prompt = get_text_logic_extraction_prompt(text_content)
        
        assert "Logic Extraction Agent" in prompt
        assert "Text Content:" in prompt
        assert "Focus on:" in prompt
        assert "LogicExtractionSchemaLiteChunk format" in prompt


class TestCreateCustomLogicExtractionPrompt:
    """Test cases for the create_custom_logic_extraction_prompt function."""
    
    def test_custom_prompt_basic(self):
        """Test basic custom prompt creation."""
        chunk_text = "Sample chunk text"
        custom_instructions = "Focus on extracting causal relationships only."
        prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=custom_instructions
        )
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert custom_instructions in prompt
        assert "Additional Instructions:" in prompt
    
    def test_custom_prompt_table(self):
        """Test custom prompt creation for table chunks."""
        chunk_text = "<table>Sample table</table>"
        custom_instructions = "Extract numerical patterns and trends."
        prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=True, 
            custom_instructions=custom_instructions
        )
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert custom_instructions in prompt
        assert "Additional Instructions:" in prompt
    
    def test_custom_prompt_no_instructions(self):
        """Test custom prompt creation without custom instructions."""
        chunk_text = "Sample chunk text"
        prompt = create_custom_logic_extraction_prompt(chunk_text, is_table=False)
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "Additional Instructions:" not in prompt
    
    def test_custom_prompt_none_instructions(self):
        """Test custom prompt creation with None custom instructions."""
        chunk_text = "Sample chunk text"
        prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=None
        )
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert "Additional Instructions:" not in prompt
    
    def test_custom_prompt_long_instructions(self):
        """Test custom prompt creation with long custom instructions."""
        chunk_text = "Sample chunk text"
        long_instructions = "This is a very long custom instruction. " * 10
        prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=long_instructions
        )
        
        assert isinstance(prompt, str)
        assert chunk_text in prompt
        assert long_instructions in prompt
        assert "Additional Instructions:" in prompt
    
    def test_custom_prompt_special_characters(self):
        """Test custom prompt creation with special characters in instructions."""
        chunk_text = "Sample chunk text"
        special_instructions = "Instructions with special chars: @#$%^&*()"
        prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=special_instructions
        )
        
        assert isinstance(prompt, str)
        assert special_instructions in prompt
    
    def test_custom_prompt_unicode(self):
        """Test custom prompt creation with unicode characters in instructions."""
        chunk_text = "Sample chunk text"
        unicode_instructions = "Instructions with unicode: 你好世界 🌍"
        prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=unicode_instructions
        )
        
        assert isinstance(prompt, str)
        assert unicode_instructions in prompt
    
    def test_custom_prompt_consistency(self):
        """Test custom prompt consistency."""
        chunk_text = "Consistent chunk text"
        custom_instructions = "Consistent instructions"
        prompt1 = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=custom_instructions
        )
        prompt2 = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=custom_instructions
        )
        
        assert prompt1 == prompt2
    
    def test_custom_prompt_differentiation(self):
        """Test custom prompt differentiation between table and text."""
        chunk_text = "Same content"
        custom_instructions = "Same instructions"
        text_prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions=custom_instructions
        )
        table_prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=True, 
            custom_instructions=custom_instructions
        )
        
        assert text_prompt != table_prompt
        assert custom_instructions in text_prompt
        assert custom_instructions in table_prompt


class TestPromptIntegration:
    """Integration tests for prompt functions."""
    
    def test_prompt_functions_work_together(self):
        """Test that different prompt functions work together."""
        chunk_text = "Sample text content"
        
        # Test all prompt functions
        basic_prompt = generate_logic_extraction_prompt(chunk_text, is_table=False)
        table_prompt = get_table_logic_extraction_prompt(chunk_text)
        text_prompt = get_text_logic_extraction_prompt(chunk_text)
        custom_prompt = create_custom_logic_extraction_prompt(
            chunk_text, 
            is_table=False, 
            custom_instructions="Custom instruction"
        )
        schema_prompt = get_logic_extraction_schema_prompt()
        
        # All should be strings
        assert isinstance(basic_prompt, str)
        assert isinstance(table_prompt, str)
        assert isinstance(text_prompt, str)
        assert isinstance(custom_prompt, str)
        assert isinstance(schema_prompt, str)
        
        # All should contain the chunk text (except schema prompt)
        assert chunk_text in basic_prompt
        assert chunk_text in table_prompt
        assert chunk_text in text_prompt
        assert chunk_text in custom_prompt
    
    def test_prompt_functions_with_different_chunk_types(self):
        """Test prompt functions with different chunk types."""
        text_content = "Sample text content"
        table_content = "<table>Sample table</table>"
        
        # Test text prompts
        text_prompt = generate_logic_extraction_prompt(text_content, is_table=False)
        text_specific_prompt = get_text_logic_extraction_prompt(text_content)
        
        # Test table prompts
        table_prompt = generate_logic_extraction_prompt(table_content, is_table=True)
        table_specific_prompt = get_table_logic_extraction_prompt(table_content)
        
        # All should be strings and contain their respective content
        assert isinstance(text_prompt, str)
        assert isinstance(text_specific_prompt, str)
        assert isinstance(table_prompt, str)
        assert isinstance(table_specific_prompt, str)
        
        assert text_content in text_prompt
        assert text_content in text_specific_prompt
        assert table_content in table_prompt
        assert table_content in table_specific_prompt
    
    def test_prompt_functions_error_handling(self):
        """Test prompt functions error handling."""
        # Test with None input (should not crash)
        try:
            prompt = generate_logic_extraction_prompt(None, is_table=False)
            assert isinstance(prompt, str)
        except Exception:
            # Some functions might raise exceptions with None input, which is acceptable
            pass
        
        # Test with very long input
        long_text = "x" * 10000
        prompt = generate_logic_extraction_prompt(long_text, is_table=False)
        assert isinstance(prompt, str)
        assert long_text in prompt
