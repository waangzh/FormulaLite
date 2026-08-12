import pytest

from formulalite.data.normalizer import normalize


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"x + y = 2", "x + y = 2"),
        (r"{{x^{2}}_{i}}", "{ { x ^ { 2 } } _ { i } }"),
        (r"\frac{a}{\frac{b}{c}}", r"\frac { a } { \frac { b } { c } }"),
        (r"\sqrt[3]{x}+\sqrt{y}", r"\sqrt [ 3 ] { x } + \sqrt { y }"),
        (
            r"\begin{array}{cc}a&b\\c&d\end{array}",
            r"\begin{array} { c c } a & b \\ c & d \end{array}",
        ),
        (
            r"\begin{matrix}a&b\\c&d\end{matrix}",
            r"\begin { m a t r i x } a & b \\ c & d \end { m a t r i x }",
        ),
        (r"\alpha+\beta=\Gamma", r"\alpha + \beta = \Gamma"),
        (r"x_i^2+x_{i+1}^{n}", r"x _ i ^ 2 + x _ { i + 1 } ^ { n }"),
    ],
)
def test_normalize_curated_latex(source: str, expected: str) -> None:
    assert normalize(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        r"x + y = 2",
        r"{{x^{2}}_{i}}",
        r"\frac{a}{\frac{b}{c}}",
        r"\sqrt[3]{x}+\sqrt{y}",
        r"\begin{array}{cc}a&b\\c&d\end{array}",
        r"\begin{matrix}a&b\\c&d\end{matrix}",
        r"\alpha+\beta=\Gamma",
    ],
)
def test_normalizer_is_idempotent(source: str) -> None:
    once = normalize(source)
    assert normalize(once) == once


def test_normalizer_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize(None)  # type: ignore[arg-type]
