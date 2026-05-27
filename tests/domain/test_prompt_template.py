"""PromptTemplate — variable extraction + rendering.

Pure value-object tests. The MCP spec's ``prompts/get`` contract
depends on these being correct: the spec lets the client send only
a subset of arguments, and unknown variables must remain as their
literal ``{{name}}`` form in the output so the client can spot
omissions.
"""
from __future__ import annotations

from src.domain.bots.value_objects import PromptTemplate


class TestVariableExtraction:
    def test_no_variables(self) -> None:
        assert PromptTemplate("Hello, world!").variables == ()

    def test_single_variable(self) -> None:
        vars_ = PromptTemplate("Hi {{name}}!").variables
        assert len(vars_) == 1
        assert vars_[0].name == "name"

    def test_multiple_distinct_variables(self) -> None:
        names = [v.name for v in PromptTemplate(
            "Hi {{first}} {{last}}. Role: {{role}}.").variables]
        assert names == ["first", "last", "role"]

    def test_duplicate_variables_deduplicated(self) -> None:
        names = [v.name for v in PromptTemplate(
            "{{n}}, {{n}}, {{n}}").variables]
        assert names == ["n"]

    def test_whitespace_tolerance(self) -> None:
        names = [v.name for v in PromptTemplate(
            "{{  name  }} and {{role}}").variables]
        assert names == ["name", "role"]


class TestRendering:
    def test_render_no_arguments_returns_original(self) -> None:
        t = PromptTemplate("Hi {{name}}")
        assert t.render({}) == "Hi {{name}}"

    def test_render_substitutes_known(self) -> None:
        t = PromptTemplate("Hi {{name}}")
        assert t.render({"name": "Vivek"}) == "Hi Vivek"

    def test_render_leaves_unknown_intact(self) -> None:
        # MCP spec lets the client supply only some args; unknown
        # variables remain so the client can spot what was missed.
        t = PromptTemplate("Hi {{name}}, age {{age}}")
        assert t.render({"name": "Vivek"}) == "Hi Vivek, age {{age}}"

    def test_render_handles_multiple_substitutions(self) -> None:
        t = PromptTemplate("{{a}}-{{b}}-{{a}}")
        assert t.render({"a": "x", "b": "y"}) == "x-y-x"

    def test_render_stringifies_values(self) -> None:
        t = PromptTemplate("Page {{page}}")
        assert t.render({"page": 42}) == "Page 42"
