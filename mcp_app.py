"""MCP App."""

from fastmcp import FastMCP


mcp = FastMCP("MCP App")


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of the two numbers
    """
    return a + b


@mcp.tool()
def subtract_numbers(a: float, b: float) -> float:
    """Subtract two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The difference of the two numbers
    """
    return a - b


@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of the two numbers
    """
    return a * b


@mcp.tool()
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The quotient of the two numbers
    """
    return a / b


if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)
