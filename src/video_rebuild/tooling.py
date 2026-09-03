from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class ToolSpec(BaseModel):
    name: str
    path: Path


class ToolCapability(BaseModel):
    name: str
    path: Path
    available: bool
    reason: str | None = None


def probe_tool(spec: ToolSpec) -> ToolCapability:
    if not spec.path.is_file():
        return ToolCapability(
            name=spec.name,
            path=spec.path,
            available=False,
            reason="path_not_found",
        )
    return ToolCapability(name=spec.name, path=spec.path, available=True)

