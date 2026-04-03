from __future__ import annotations

from bot.formatting import markdown_to_telegram_html


class TestBasicFormatting:
    def test_bold(self) -> None:
        result = markdown_to_telegram_html("**bold text**")
        assert "<strong>bold text</strong>" in result

    def test_italic(self) -> None:
        result = markdown_to_telegram_html("*italic text*")
        assert "<em>italic text</em>" in result

    def test_inline_code(self) -> None:
        result = markdown_to_telegram_html("`inline code`")
        assert "<code>inline code</code>" in result

    def test_link(self) -> None:
        result = markdown_to_telegram_html("[click here](https://example.com)")
        assert '<a href="https://example.com">click here</a>' in result

    def test_code_block(self) -> None:
        result = markdown_to_telegram_html("```python\nprint('hi')\n```")
        assert "<pre>" in result

    def test_heading_becomes_bold(self) -> None:
        result = markdown_to_telegram_html("# Heading 1")
        assert "<strong>Heading 1</strong>" in result


class TestOrderedList:
    def test_converts_ordered_list_to_bullets(self) -> None:
        md = "1. first\n2. second\n3. third"
        result = markdown_to_telegram_html(md)
        assert "1. first" not in result
        assert "2. second" not in result
        assert "3. third" not in result
        assert "- first" in result
        assert "- second" in result
        assert "- third" in result

    def test_unordered_list_unchanged(self) -> None:
        md = "- apple\n- banana"
        result = markdown_to_telegram_html(md)
        assert "- apple" in result
        assert "- banana" in result

    def test_complex_ordered_list_falls_back_to_bullets(self) -> None:
        md = "1. OpenAI SDK\n   - time\n   - point\n2. MCP protocol\n   - time\n   - point"
        result = markdown_to_telegram_html(md)
        assert "1. OpenAI SDK" not in result
        assert "2. MCP protocol" not in result
        assert "- OpenAI SDK" in result
        assert "- MCP protocol" in result


class TestHtmlEscaping:
    def test_special_chars_escaped(self) -> None:
        result = markdown_to_telegram_html("a < b & c > d")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result


class TestPlainText:
    def test_plain_text_unchanged(self) -> None:
        result = markdown_to_telegram_html("just plain text")
        assert "just plain text" in result

    def test_strips_trailing_whitespace(self) -> None:
        result = markdown_to_telegram_html("hello")
        assert not result.endswith("\n\n")
