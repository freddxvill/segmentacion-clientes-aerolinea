"""Formato de cifras en español para el texto del sitio (cifras inline en los `.qmd`)."""


def num(value: float, decimals: int = 0) -> str:
    """12345.6 → '12.345,6' (separador de miles '.', decimal ',')."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def pct(value: float, decimals: int = 1) -> str:
    """0.3171 → '31,7 %'."""
    return f"{num(value * 100, decimals)} %"
