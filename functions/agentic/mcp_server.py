#!/usr/bin/env python3
"""
MCP Server CLI Tool

Launch an MCP server with file system and data analysis tools.

Examples:
    python mcp_server.py "List all files in the workspace"
    python mcp_server.py "Create a file called notes.txt with content: Hello World"
    python mcp_server.py "Calculate statistics for: 10, 20, 30, 40, 50" --model large

Example Output:
    $ python mcp_server.py "List all files in the workspace"

    Loading model: llama3.2:1b
    Initializing MCP server...
    Tools available: read_file, write_file, list_files, delete_file, calculate_statistics

    ======================================================================
    USER QUERY
    ======================================================================
    List all files in the workspace

    ======================================================================
    AGENT EXECUTION
    ======================================================================

    Tool Call: list_files()
    Tool Result: ['notes.txt', 'data.csv', 'report.md']

    Agent Response: The workspace contains 3 files: notes.txt, data.csv, and report.md.

    ======================================================================

    $ python mcp_server.py "Create a file called greeting.txt with: Hello from MCP!"

    ======================================================================
    AGENT EXECUTION
    ======================================================================

    Tool Call: write_file(filename='greeting.txt', content='Hello from MCP!')
    Tool Result: File written successfully

    Agent Response: I've created the file 'greeting.txt' with the content "Hello from MCP!".

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


class MCPServer:
    """MCP Server with file system and data analysis tools."""

    def __init__(self, workspace_path='./mcp_workspace'):
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(exist_ok=True)

        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'read_file',
                    'description': 'Read contents of a file',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'filename': {'type': 'string', 'description': 'Name of file to read'}
                        },
                        'required': ['filename']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'write_file',
                    'description': 'Write content to a file',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'filename': {'type': 'string', 'description': 'Name of file to write'},
                            'content': {'type': 'string', 'description': 'Content to write'}
                        },
                        'required': ['filename', 'content']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_files',
                    'description': 'List all files in workspace',
                    'parameters': {'type': 'object', 'properties': {}}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'delete_file',
                    'description': 'Delete a file',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'filename': {'type': 'string', 'description': 'Name of file to delete'}
                        },
                        'required': ['filename']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'calculate_statistics',
                    'description': 'Calculate statistics (mean, median, min, max) for a list of numbers',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'numbers': {
                                'type': 'array',
                                'items': {'type': 'number'},
                                'description': 'List of numbers'
                            }
                        },
                        'required': ['numbers']
                    }
                }
            }
        ]

        self.function_map = {
            'read_file': self.read_file,
            'write_file': self.write_file,
            'list_files': self.list_files,
            'delete_file': self.delete_file,
            'calculate_statistics': self.calculate_statistics
        }

    def _safe_path(self, filename):
        """Ensure path is within workspace."""
        path = (self.workspace / filename).resolve()
        if not str(path).startswith(str(self.workspace.resolve())):
            raise ValueError("Path outside workspace")
        return path

    def read_file(self, filename):
        """Read file contents."""
        path = self._safe_path(filename)
        if not path.exists():
            return f"Error: File '{filename}' not found"
        return path.read_text()

    def write_file(self, filename, content):
        """Write content to file."""
        path = self._safe_path(filename)
        path.write_text(content)
        return "File written successfully"

    def list_files(self):
        """List all files in workspace."""
        files = [f.name for f in self.workspace.iterdir() if f.is_file()]
        return files if files else "No files found"

    def delete_file(self, filename):
        """Delete a file."""
        path = self._safe_path(filename)
        if not path.exists():
            return f"Error: File '{filename}' not found"
        path.unlink()
        return "File deleted successfully"

    def calculate_statistics(self, numbers):
        """Calculate statistics for numbers."""
        if not numbers:
            return "Error: Empty list"
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        median = sorted_nums[n//2] if n % 2 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2
        return {
            'count': n,
            'mean': sum(numbers) / n,
            'median': median,
            'min': min(numbers),
            'max': max(numbers)
        }


def run_server(query, model='llama3.2:1b', workspace='./mcp_workspace', max_iterations=5):
    """
    Run MCP server and execute query.

    Args:
        query: User query
        model: Ollama model name
        workspace: Workspace directory path
        max_iterations: Maximum tool-calling iterations

    Returns:
        Final answer from agent
    """
    server = MCPServer(workspace)

    messages = [{'role': 'user', 'content': query}]

    print("\n" + "="*70)
    print("AGENT EXECUTION")
    print("="*70 + "\n")

    for iteration in range(max_iterations):
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=server.tools
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

            print(f"Tool Call: {function_name}({', '.join(f'{k}={repr(v)}' for k, v in function_args.items()) if function_args else ''})")

            function_to_call = server.function_map[function_name]
            function_response = function_to_call(**function_args)

            print(f"Tool Result: {function_response}\n")

            messages.append({
                'role': 'tool',
                'content': str(function_response)
            })

    return "Maximum iterations reached"


def main():
    parser = argparse.ArgumentParser(
        description='Run an MCP server with file system and data analysis tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "List all files in the workspace"
  %(prog)s "Create a file called notes.txt with: Hello World"
  %(prog)s "Read the file notes.txt"
  %(prog)s "Calculate statistics for: 10, 20, 30, 40, 50"

Available tools:
  - read_file: Read file contents
  - write_file: Create or update files
  - list_files: List workspace files
  - delete_file: Delete files
  - calculate_statistics: Compute mean, median, min, max

Workspace: All file operations are scoped to ./mcp_workspace/
        """
    )

    parser.add_argument('query', type=str, help='Query for the MCP server')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (llama3.2:1b), large (llama3.1:8b)')
    parser.add_argument('--workspace', type=str, default='./mcp_workspace',
                        help='Workspace directory path')
    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum number of tool-calling iterations')

    args = parser.parse_args()

    models = {
        'small': 'llama3.2:1b',
        'large': 'llama3.1:8b'
    }
    model_name = models[args.model]

    print(f"Loading model: {model_name}")
    print(f"Initializing MCP server...")
    print("Tools available: read_file, write_file, list_files, delete_file, calculate_statistics")

    print("\n" + "="*70)
    print("USER QUERY")
    print("="*70)
    print(args.query)

    try:
        result = run_server(
            args.query,
            model=model_name,
            workspace=args.workspace,
            max_iterations=args.max_iterations
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if "connection" in str(e).lower():
            print("Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
