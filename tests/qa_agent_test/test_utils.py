"""
Unit tests for the QA agent utility functions.
"""

import pytest
from qa_agent.utils import (
    format_answer,
    extract_sources,
    validate_question,
    create_session,
    cleanup_session,
    extract_keywords,
    calculate_answer_quality,
    format_conversation_history,
    sanitize_input
)
from qa_agent.exceptions import ValidationError, SessionError


class TestFormatAnswer:
    """Test cases for the format_answer function."""

    def test_format_answer_basic(self):
        """Test basic answer formatting."""
        answer = "This is a test answer."
        formatted = format_answer(answer)
        assert formatted == "This is a test answer."

    def test_format_answer_concise_style(self):
        """Test concise style formatting."""
        answer = "This is a very long answer with lots of unnecessary words and redundant information that could be made more concise."
        formatted = format_answer(answer, style="concise")
        assert len(formatted) <= len(answer)
        assert "This is a very long answer" in formatted

    def test_format_answer_detailed_style(self):
        """Test detailed style formatting."""
        answer = "Short answer."
        formatted = format_answer(answer, style="detailed")
        assert "Based on the available information:" in formatted

    def test_format_answer_academic_style(self):
        """Test academic style formatting."""
        answer = "This is a statement without proper academic formatting"
        formatted = format_answer(answer, style="academic")
        assert formatted.endswith('.')
        assert "According to the available information" in formatted

    def test_format_answer_max_length(self):
        """Test answer length truncation."""
        long_answer = "This is a very long answer. " * 100
        formatted = format_answer(long_answer, max_length=100)
        assert len(formatted) <= 103  # 100 + "..."
        assert formatted.endswith("...")

    def test_format_answer_empty(self):
        """Test formatting empty answer."""
        formatted = format_answer("")
        assert "I apologize" in formatted

    def test_format_answer_none(self):
        """Test formatting None answer."""
        formatted = format_answer(None)
        assert "I apologize" in formatted


class TestExtractSources:
    """Test cases for the extract_sources function."""

    def test_extract_sources_numeric_citations(self):
        """Test extraction of numeric citations."""
        answer = "According to research [1], the results show [2] that this is true."
        sources = extract_sources(answer)
        assert len(sources) == 2
        assert sources[0]["text"] == "1"
        assert sources[1]["text"] == "2"

    def test_extract_sources_parenthetical_citations(self):
        """Test extraction of parenthetical citations."""
        answer = "The study (Smith, 2023) shows that (Jones et al., 2022) this is true."
        sources = extract_sources(answer)
        assert len(sources) >= 2
        assert "Smith, 2023" in [s["text"] for s in sources]

    def test_extract_sources_source_prefix(self):
        """Test extraction of 'Source:' prefixed citations."""
        answer = "This is true. Source: Research Paper 2023. Another fact. Source: Journal Article 2022."
        sources = extract_sources(answer)
        assert len(sources) >= 2
        assert any("Research Paper 2023" in s["text"] for s in sources)

    def test_extract_sources_according_to(self):
        """Test extraction of 'According to' citations."""
        answer = "According to the research paper, this is true."
        sources = extract_sources(answer)
        assert len(sources) >= 1
        assert any("the research paper" in s["text"] for s in sources)

    def test_extract_sources_no_citations(self):
        """Test extraction when no citations are present."""
        answer = "This is just a regular answer without any citations."
        sources = extract_sources(answer)
        assert len(sources) == 0

    def test_extract_sources_empty_answer(self):
        """Test extraction from empty answer."""
        sources = extract_sources("")
        assert len(sources) == 0


class TestValidateQuestion:
    """Test cases for the validate_question function."""

    def test_validate_question_valid(self):
        """Test validation of valid questions."""
        valid_questions = [
            "What is machine learning?",
            "How does the attention mechanism work?",
            "Explain transformers in detail."
        ]
        for question in valid_questions:
            assert validate_question(question) is True

    def test_validate_question_empty(self):
        """Test validation of empty questions."""
        with pytest.raises(ValidationError, match="Question cannot be empty"):
            validate_question("")

    def test_validate_question_none(self):
        """Test validation of None questions."""
        with pytest.raises(ValidationError, match="Question cannot be empty"):
            validate_question(None)

    def test_validate_question_too_short(self):
        """Test validation of questions that are too short."""
        with pytest.raises(ValidationError, match="Question must be at least 3 characters long"):
            validate_question("a")

    def test_validate_question_too_long(self):
        """Test validation of questions that are too long."""
        long_question = "What is machine learning? " * 100  # Over 1000 characters
        with pytest.raises(ValidationError, match="Question is too long"):
            validate_question(long_question)

    def test_validate_question_not_string(self):
        """Test validation of non-string questions."""
        with pytest.raises(ValidationError, match="Question must be a string"):
            validate_question(123)

    def test_validate_question_harmful_content(self):
        """Test validation of questions with potentially harmful content."""
        harmful_questions = [
            "What is <script>alert('xss')</script>?",
            "Tell me about javascript:alert('xss')",
            "Explain eval('malicious code')"
        ]
        for question in harmful_questions:
            with pytest.raises(ValidationError, match="Question contains potentially harmful content"):
                validate_question(question)


class TestCreateSession:
    """Test cases for the create_session function."""

    def test_create_session_returns_string(self):
        """Test that create_session returns a string."""
        session_id = create_session()
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_create_session_unique(self):
        """Test that create_session returns unique IDs."""
        session_ids = [create_session() for _ in range(10)]
        assert len(set(session_ids)) == 10  # All unique


class TestCleanupSession:
    """Test cases for the cleanup_session function."""

    def test_cleanup_session_success(self):
        """Test successful session cleanup."""
        session_id = "test_session_123"
        # Should not raise an exception
        cleanup_session(session_id)

    def test_cleanup_session_invalid_id(self):
        """Test cleanup with invalid session ID."""
        # Should not raise an exception even with invalid ID
        cleanup_session("invalid_session")


class TestExtractKeywords:
    """Test cases for the extract_keywords function."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        question = "What is machine learning and how does it work?"
        keywords = extract_keywords(question)
        assert "machine" in keywords
        assert "learning" in keywords
        assert "work" in keywords
        assert "what" not in keywords  # Stop word removed

    def test_extract_keywords_stop_words(self):
        """Test that stop words are removed."""
        question = "The and or but what when where why how"
        keywords = extract_keywords(question)
        assert len(keywords) == 0  # All stop words

    def test_extract_keywords_short_words(self):
        """Test that short words are removed."""
        question = "a b c machine learning"
        keywords = extract_keywords(question)
        assert "machine" in keywords
        assert "learning" in keywords
        assert "a" not in keywords
        assert "b" not in keywords
        assert "c" not in keywords

    def test_extract_keywords_empty(self):
        """Test keyword extraction from empty question."""
        keywords = extract_keywords("")
        assert len(keywords) == 0


class TestCalculateAnswerQuality:
    """Test cases for the calculate_answer_quality function."""

    def test_calculate_quality_basic(self):
        """Test basic quality calculation."""
        answer = "Machine learning is a subset of artificial intelligence."
        question = "What is machine learning?"
        quality = calculate_answer_quality(answer, question)
        
        assert "quality_score" in quality
        assert "metrics" in quality
        assert 0.0 <= quality["quality_score"] <= 1.0
        assert quality["metrics"]["answer_length"] > 0

    def test_calculate_quality_with_sources(self):
        """Test quality calculation with sources."""
        answer = "According to research [1], machine learning is important."
        question = "What is machine learning?"
        quality = calculate_answer_quality(answer, question)
        
        assert quality["metrics"]["has_sources"] is True
        assert quality["quality_score"] > 0.5

    def test_calculate_quality_empty_answer(self):
        """Test quality calculation with empty answer."""
        quality = calculate_answer_quality("", "What is machine learning?")
        assert quality["quality_score"] == 0.0

    def test_calculate_quality_question_back(self):
        """Test quality calculation when answer is just a question."""
        answer = "What do you mean?"
        question = "What is machine learning?"
        quality = calculate_answer_quality(answer, question)
        
        # Should have lower quality score
        assert quality["quality_score"] < 0.5


class TestFormatConversationHistory:
    """Test cases for the format_conversation_history function."""

    def test_format_history_basic(self):
        """Test basic conversation history formatting."""
        messages = [
            {"type": "HumanMessage", "content": "What is AI?"},
            {"type": "AIMessage", "content": "AI is artificial intelligence."}
        ]
        formatted = format_conversation_history(messages)
        
        assert "1. User: What is AI?" in formatted
        assert "2. Assistant: AI is artificial intelligence." in formatted

    def test_format_history_empty(self):
        """Test formatting empty conversation history."""
        formatted = format_conversation_history([])
        assert "No conversation history available." in formatted

    def test_format_history_with_timestamps(self):
        """Test formatting with timestamps."""
        messages = [
            {
                "type": "HumanMessage", 
                "content": "What is AI?",
                "timestamp": "2024-01-01T12:00:00"
            }
        ]
        formatted = format_conversation_history(messages)
        assert "1. User: What is AI?" in formatted


class TestSanitizeInput:
    """Test cases for the sanitize_input function."""

    def test_sanitize_input_basic(self):
        """Test basic input sanitization."""
        text = "What is machine learning?"
        sanitized = sanitize_input(text)
        assert sanitized == "What is machine learning?"

    def test_sanitize_input_harmful_chars(self):
        """Test sanitization of harmful characters."""
        text = "What is <script>alert('xss')</script> machine learning?"
        sanitized = sanitize_input(text)
        assert "<script>" not in sanitized
        assert "alert" not in sanitized

    def test_sanitize_input_excessive_whitespace(self):
        """Test sanitization of excessive whitespace."""
        text = "What    is    machine    learning?"
        sanitized = sanitize_input(text)
        assert "    " not in sanitized
        assert " " in sanitized  # Single spaces should remain

    def test_sanitize_input_too_long(self):
        """Test sanitization of overly long input."""
        text = "What is machine learning? " * 100  # Very long
        sanitized = sanitize_input(text)
        assert len(sanitized) <= 2003  # 2000 + "..."
        assert sanitized.endswith("...")

    def test_sanitize_input_empty(self):
        """Test sanitization of empty input."""
        sanitized = sanitize_input("")
        assert sanitized == ""

    def test_sanitize_input_none(self):
        """Test sanitization of None input."""
        sanitized = sanitize_input(None)
        assert sanitized == ""
