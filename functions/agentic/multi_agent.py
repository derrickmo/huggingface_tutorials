#!/usr/bin/env python3
"""
Multi-Agent CLI Tool

Run agents with different patterns (ReAct, Plan-and-Execute, Reflection) using multiple tools.

Examples:
    python multi_agent.py "Find files starting with 'data' and calculate their count" --pattern react
    python multi_agent.py "Analyze the number 42: is it prime? what are its factors?" --pattern plan
    python multi_agent.py "Calculate 100 / 4 and verify the result" --pattern reflection

Example Output:
    $ python multi_agent.py "Find files in workspace and count them" --pattern react

    Loading model: llama3.2:1b
    Agent pattern: ReAct (Reason + Act)
    Tools: 7 available

    ======================================================================
    USER TASK
    ======================================================================
    Find files in workspace and count them

    ======================================================================
    AGENT EXECUTION (ReAct)
    ======================================================================

    Tool Call: list_files()
    Tool Result: ['notes.txt', 'data.csv', 'report.md']

    Tool Call: count_items(items=['notes.txt', 'data.csv', 'report.md'])
    Tool Result: {'count': 3, 'items': ['notes.txt', 'data.csv', 'report.md']}

    Agent Response: There are 3 files in the workspace: notes.txt, data.csv, and report.md.

    ======================================================================

    $ python multi_agent.py "Check if 17 is prime and calculate 17 * 3" --pattern plan

    ======================================================================
    AGENT EXECUTION (Plan-and-Execute)
    ======================================================================

    Phase 1: Planning
    Plan:
    1. Check if 17 is a prime number
    2. Calculate 17 multiplied by 3

    Phase 2: Execution
    Tool Call: is_prime(number=17)
    Tool Result: True

    Tool Call: calculator(operation='multiply', a=17, b=3)
    Tool Result: 51

    Agent Response: Yes, 17 is a prime number. When multiplied by 3, the result is 51.

    ======================================================================
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import ollama
except ImportError:
    print("Error: 'ollama' package not found.", file=sys.stderr)
    print("Install with: pip install ollama", file=sys.stderr)
    sys.exit(1)


class ComprehensiveToolServer:
    """Multi-tool server with file ops, math, text, and data analysis."""

    def __init__(self, workspace_path='./mcp_workspace'):
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(exist_ok=True)

        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'calculator',
                    'description': 'Perform arithmetic operations',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'operation': {'type': 'string', 'enum': ['add', 'subtract', 'multiply', 'divide']},
                            'a': {'type': 'number'},
                            'b': {'type': 'number'}
                        },
                        'required': ['operation', 'a', 'b']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'is_prime',
                    'description': 'Check if a number is prime',
                    'parameters': {
                        'type': 'object',
                        'properties': {'number': {'type': 'integer'}},
                        'required': ['number']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'factorize',
                    'description': 'Find all factors of a number',
                    'parameters': {
                        'type': 'object',
                        'properties': {'number': {'type': 'integer'}},
                        'required': ['number']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_files',
                    'description': 'List files in workspace',
                    'parameters': {'type': 'object', 'properties': {}}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'search_files',
                    'description': 'Find files matching a pattern',
                    'parameters': {
                        'type': 'object',
                        'properties': {'pattern': {'type': 'string'}},
                        'required': ['pattern']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'count_items',
                    'description': 'Count items in a list',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'items': {'type': 'array', 'items': {'type': 'string'}}
                        },
                        'required': ['items']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'reverse_text',
                    'description': 'Reverse a text string',
                    'parameters': {
                        'type': 'object',
                        'properties': {'text': {'type': 'string'}},
                        'required': ['text']
                    }
                }
            }
        ]

        self.function_map = {
            'calculator': self.calculator,
            'is_prime': self.is_prime,
            'factorize': self.factorize,
            'list_files': self.list_files,
            'search_files': self.search_files,
            'count_items': self.count_items,
            'reverse_text': self.reverse_text
        }

    def calculator(self, operation, a, b):
        """Perform arithmetic."""
        ops = {'add': a + b, 'subtract': a - b, 'multiply': a * b, 'divide': a / b if b != 0 else "Error: Division by zero"}
        return ops.get(operation, "Unknown operation")

    def is_prime(self, number):
        """Check if number is prime."""
        if number < 2:
            return False
        for i in range(2, int(number ** 0.5) + 1):
            if number % i == 0:
                return False
        return True

    def factorize(self, number):
        """Find all factors."""
        factors = [i for i in range(1, abs(number) + 1) if number % i == 0]
        return factors

    def list_files(self):
        """List workspace files."""
        files = [f.name for f in self.workspace.iterdir() if f.is_file()]
        return files if files else []

    def search_files(self, pattern):
        """Find files matching pattern."""
        files = [f.name for f in self.workspace.iterdir() if f.is_file() and pattern.lower() in f.name.lower()]
        return files if files else []

    def count_items(self, items):
        """Count items in list."""
        return {'count': len(items), 'items': items}

    def reverse_text(self, text):
        """Reverse text."""
        return text[::-1]


def run_react_agent(query, model, server, max_iterations=5):
    """ReAct pattern: Reason and Act iteratively."""
    messages = [{'role': 'user', 'content': query}]

    print("\n" + "="*70)
    print("AGENT EXECUTION (ReAct)")
    print("="*70 + "\n")

    for iteration in range(max_iterations):
        response = ollama.chat(model=model, messages=messages, tools=server.tools)
        messages.append(response['message'])

        if not response['message'].get('tool_calls'):
            print(f"Agent Response: {response['message']['content']}")
            print("\n" + "="*70)
            return response['message']['content']

        for tool_call in response['message']['tool_calls']:
            func_name = tool_call['function']['name']
            func_args = tool_call['function']['arguments']
            print(f"Tool Call: {func_name}({', '.join(f'{k}={repr(v)}' for k, v in func_args.items()) if func_args else ''})")

            result = server.function_map[func_name](**func_args)
            print(f"Tool Result: {result}\n")

            messages.append({'role': 'tool', 'content': str(result)})

    return "Maximum iterations reached"


def run_plan_execute_agent(query, model, server):
    """Plan-and-Execute pattern: Plan first, then execute."""
    print("\n" + "="*70)
    print("AGENT EXECUTION (Plan-and-Execute)")
    print("="*70 + "\n")

    # Phase 1: Planning
    print("Phase 1: Planning")
    plan_prompt = f"""Create a step-by-step plan to accomplish this task: {query}

Available tools: {', '.join([t['function']['name'] for t in server.tools])}

List the plan as numbered steps."""

    plan_response = ollama.chat(model=model, messages=[{'role': 'user', 'content': plan_prompt}])
    plan = plan_response['message']['content']
    print(f"Plan:\n{plan}\n")

    # Phase 2: Execution
    print("Phase 2: Execution")
    exec_prompt = f"""Execute this plan step by step: {plan}

Original task: {query}"""

    return run_react_agent(exec_prompt, model, server, max_iterations=5)


def run_reflection_agent(query, model, server):
    """Reflection pattern: Execute, then verify results."""
    print("\n" + "="*70)
    print("AGENT EXECUTION (Reflection)")
    print("="*70 + "\n")

    # Initial execution
    print("Phase 1: Initial Execution")
    initial_result = run_react_agent(query, model, server, max_iterations=3)

    # Reflection
    print("\nPhase 2: Reflection and Verification")
    reflect_prompt = f"""Review this result and verify its correctness: {initial_result}

Original task: {query}

If correct, confirm it. If incorrect, provide the correct answer."""

    return run_react_agent(reflect_prompt, model, server, max_iterations=3)


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-tool agents with different patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Find files starting with 'data' and count them" --pattern react
  %(prog)s "Check if 17 is prime and calculate 17 * 3" --pattern plan
  %(prog)s "Calculate 144 / 12 and verify the result" --pattern reflection

Agent Patterns:
  react: Reason and Act iteratively (fastest)
  plan: Plan-and-Execute (structured)
  reflection: Execute then verify (most thorough)

Available Tools (7):
  - calculator, is_prime, factorize
  - list_files, search_files, count_items
  - reverse_text
        """
    )

    parser.add_argument('query', type=str, help='Task for the agent')
    parser.add_argument('--pattern', choices=['react', 'plan', 'reflection'],
                        default='react', help='Agent pattern to use')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (llama3.2:1b), large (llama3.1:8b)')
    parser.add_argument('--workspace', type=str, default='./mcp_workspace',
                        help='Workspace directory path')

    args = parser.parse_args()

    models = {
        'small': 'llama3.2:1b',
        'large': 'llama3.1:8b'
    }
    model_name = models[args.model]

    pattern_names = {
        'react': 'ReAct (Reason + Act)',
        'plan': 'Plan-and-Execute',
        'reflection': 'Reflection and Verification'
    }

    print(f"Loading model: {model_name}")
    print(f"Agent pattern: {pattern_names[args.pattern]}")

    server = ComprehensiveToolServer(args.workspace)
    print(f"Tools: {len(server.tools)} available")

    print("\n" + "="*70)
    print("USER TASK")
    print("="*70)
    print(args.query)

    try:
        if args.pattern == 'react':
            result = run_react_agent(args.query, model_name, server)
        elif args.pattern == 'plan':
            result = run_plan_execute_agent(args.query, model_name, server)
        else:  # reflection
            result = run_reflection_agent(args.query, model_name, server)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if "connection" in str(e).lower():
            print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
