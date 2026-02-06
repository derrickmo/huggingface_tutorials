#!/usr/bin/env python3
"""
MCP Agent CLI Tool

Run a tool-using agent powered by local LLMs via Ollama.

Examples:
    python mcp_agent.py "What is 25 multiplied by 17?"
    python mcp_agent.py "Calculate (100 - 35) / 5" --model large
    python mcp_agent.py "What's 1234 + 5678?" --max-iterations 3

Example Output:
    $ python mcp_agent.py "What is 127 multiplied by 83?"

    Loading model: llama3.2:1b
    Initializing tools: calculator

    ======================================================================
    USER QUERY
    ======================================================================
    What is 127 multiplied by 83?

    ======================================================================
    AGENT EXECUTION
    ======================================================================

    Tool Call: calculator(operation='multiply', a=127, b=83)
    Tool Result: 10541

    Agent Response: 127 multiplied by 83 equals 10,541.

    ======================================================================

    $ python mcp_agent.py "If I have $100 and spend $15 three times, how much is left?"

    ======================================================================
    AGENT EXECUTION
    ======================================================================

    Tool Call: calculator(operation='multiply', a=15, b=3)
    Tool Result: 45

    Tool Call: calculator(operation='subtract', a=100, b=45)
    Tool Result: 55

    Agent Response: After spending $15 three times ($45 total), you would have $55 left from your original $100.

    ======================================================================
"""

import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import ollama
except ImportError:
    print("Error: 'ollama' package not found.", file=sys.stderr)
    print("Install with: pip install ollama", file=sys.stderr)
    print("Also install Ollama from: https://ollama.ai/", file=sys.stderr)
    sys.exit(1)


def calculator(operation: str, a: float, b: float) -> float:
    """Execute calculator operations."""
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            return "Error: Division by zero"
        return a / b
    else:
        return "Error: Unknown operation"


def run_agent(query, model='llama3.2:1b', max_iterations=5):
    """
    Run a tool-using agent to answer a query.

    Args:
        query: User question
        model: Ollama model name
        max_iterations: Maximum tool-calling iterations

    Returns:
        Final answer from agent
    """
    # Define tools
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'calculator',
                'description': 'Perform basic arithmetic operations (add, subtract, multiply, divide)',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'operation': {
                            'type': 'string',
                            'description': 'The operation to perform',
                            'enum': ['add', 'subtract', 'multiply', 'divide']
                        },
                        'a': {
                            'type': 'number',
                            'description': 'First number'
                        },
                        'b': {
                            'type': 'number',
                            'description': 'Second number'
                        }
                    },
                    'required': ['operation', 'a', 'b']
                }
            }
        }
    ]

    available_functions = {'calculator': calculator}

    messages = [{'role': 'user', 'content': query}]

    print("\n" + "="*70)
    print("AGENT EXECUTION")
    print("="*70 + "\n")

    for iteration in range(max_iterations):
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=tools
        )

        messages.append(response['message'])

        if not response['message'].get('tool_calls'):
            final_answer = response['message']['content']
            print(f"Agent Response: {final_answer}")
            print("\n" + "="*70)
            return final_answer

        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            function_args = tool_call['function']['arguments']

            print(f"Tool Call: {function_name}({', '.join(f'{k}={repr(v)}' for k, v in function_args.items())})")

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            print(f"Tool Result: {function_response}\n")

            messages.append({
                'role': 'tool',
                'content': str(function_response)
            })

    return "Maximum iterations reached"


def main():
    parser = argparse.ArgumentParser(
        description='Run a tool-using agent with local LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is 25 multiplied by 17?"
  %(prog)s "Calculate (100 - 35) / 5" --model large
  %(prog)s "What's 1234 + 5678?" --max-iterations 3

Supported models:
  small: llama3.2:1b (1.3GB, CPU-friendly)
  large: llama3.1:8b (4.7GB, GPU-optimized)

Note: Requires Ollama installed with models pulled:
  ollama pull llama3.2:1b
  ollama pull llama3.1:8b
        """
    )

    parser.add_argument('query', type=str, help='Question or task for the agent')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (llama3.2:1b), large (llama3.1:8b)')
    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum number of tool-calling iterations')

    args = parser.parse_args()

    # Map model size to Ollama model name
    models = {
        'small': 'llama3.2:1b',
        'large': 'llama3.1:8b'
    }
    model_name = models[args.model]

    print(f"Loading model: {model_name}")
    print("Initializing tools: calculator")

    print("\n" + "="*70)
    print("USER QUERY")
    print("="*70)
    print(args.query)

    try:
        result = run_agent(
            args.query,
            model=model_name,
            max_iterations=args.max_iterations
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if "connection" in str(e).lower():
            print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
