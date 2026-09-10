import json
import os
import sys

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


class MovieMCPError(RuntimeError):
    pass


class MovieMCPClient:
    async def call_tool(self, tool_name: str, arguments: dict) -> dict|list:
        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=os.environ.copy(),)

        async with Client(stdio_client(parameters)) as client:
            result = await client.call_tool(tool_name, arguments)

        text = self._extract_text(result)

        if result.is_error:
            detail = text or "unknown tool error"
            raise MovieMCPError(f"Error calling {tool_name}: {detail}")

        if not text:
            raise MovieMCPError(f"{tool_name} returned no text content")

        return json.loads(text)

    @staticmethod
    def _extract_text(result) -> str:
        text_blocks = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(text_blocks)
