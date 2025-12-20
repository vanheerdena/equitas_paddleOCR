#!/usr/bin/env python3
"""Start script for the PaddleOCR FastAPI service.

This script runs the uvicorn server with appropriate settings.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the uvicorn server in the conda environment."""
    # Get the project root directory
    project_root = Path(__file__).parent.resolve()

    # Change to project root to ensure relative imports work
    import os
    os.chdir(project_root)

    # Build uvicorn command using conda run to activate the environment
    cmd = [
        "conda",
        "run",
        "-n",
        "equitasOCR",
        "python",
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]

    print("Starting PaddleOCR FastAPI service...")
    print(f"Activating conda environment: equitasOCR")
    print(f"Command: {' '.join(cmd)}")
    print(f"Project root: {project_root}")
    print("Press CTRL+C to stop the server\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nError starting server: {e}")
        print("Make sure conda is installed and the 'equitasOCR' environment exists.")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'conda' command not found.")
        print("Make sure conda is installed and available in your PATH.")
        sys.exit(1)


if __name__ == "__main__":
    main()

