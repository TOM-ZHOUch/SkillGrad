"""Skill patch stage.

The patcher reads the current task evidence, optional momentum overlay, and
current skill files. It evolves the skill in place by writing new content via
the write_file tool. In the current pipeline every patch is accepted; skill
snapshots are kept for forensics.
"""

import time
from pathlib import Path
from typing import Optional

from agents import Agent, Runner

from pipeline.helpers import _build_file_tools, _resolve_model
from prompts.patcher import PATCHER_PROMPT
from runners.cost_tracker import CostTracker
from runners.model_settings import get_model_settings
from runners.trajectory_logger import TrajectoryLogger, stream_with_logging


async def run_patch(
    batch_diagnoses_path: Path,
    skills_dir: Path,
    model: str,
    project_root: Path,
    cost_tracker: CostTracker,
    iter_dir: Path,
    overlay_path: Optional[Path] = None,
    momentum_memory_path: Optional[Path] = None,
) -> str:
    """Run the patcher agent to evolve the skill."""
    read_file, write_file = _build_file_tools(project_root)

    agent = Agent(
        name="Patcher",
        instructions=PATCHER_PROMPT,
        model=_resolve_model(model),
        model_settings=get_model_settings(model),
        tools=[read_file, write_file],
    )

    # List actual files so the agent doesn't guess filenames
    skill_dir = skills_dir / "xlsx"
    skill_files = [str(skill_dir / "SKILL.md")]
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for f in sorted(refs_dir.iterdir()):
            if f.is_file():
                skill_files.append(str(f))

    query = (
        f"Evolve the skill based on the analysis below.\n\n"
        f"Original diagnoses: {batch_diagnoses_path}\n"
    )
    if overlay_path is not None:
        query += f"Per-attempt overlay: {overlay_path}\n"
    if momentum_memory_path is not None:
        query += f"Cross-iteration pattern record: {momentum_memory_path}\n"
    query += (
        f"Skill files:\n"
        + "\n".join(f"  - {p}" for p in skill_files)
        + "\n"
        f"Reference directory for new L3 files: {refs_dir}\n"
        f"(Create files there when a reusable procedure should live outside "
        f"`SKILL.md`; it may not exist yet.)\n"
    )

    logger = TrajectoryLogger(iter_dir / "patcher.jsonl")
    print(f"  [patch] Evolving skill")
    t0 = time.time()
    output = ""
    try:
        result = Runner.run_streamed(agent, query, max_turns=20)
        await stream_with_logging(result, logger)
        output = result.final_output or ""
        delta = cost_tracker.update(result)
        cost_tracker.print_step("PATCH", delta)
    except Exception as e:
        print(f"  [patch] error: {e}")
        output = f"[PATCH ERROR] {e}"
    logger.flush()
    elapsed = round(time.time() - t0, 2)
    print(f"  [patch] Done in {elapsed}s")

    return output
