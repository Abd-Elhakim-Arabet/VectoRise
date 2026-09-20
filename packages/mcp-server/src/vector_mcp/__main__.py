"""Run the VectoRise MCP server over stdio."""

from __future__ import annotations

import asyncio
import sys


def main() -> int:
    from vector_mcp.server import mcp

    try:
        asyncio.run(mcp.run_stdio_async())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
