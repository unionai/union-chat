"""This module contains calculator tools for an LLM to use.
"""


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a + b


def subtract_numbers(a: float, b: float) -> float:
    """Subtract two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a - b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a * b


def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a / b


if __name__ == "__main__":
    from langchain_core.utils.function_calling import convert_to_openai_tool

    print(convert_to_openai_tool(add_numbers))
    print(convert_to_openai_tool(subtract_numbers))
    print(convert_to_openai_tool(multiply_numbers))
    print(convert_to_openai_tool(divide_numbers))
