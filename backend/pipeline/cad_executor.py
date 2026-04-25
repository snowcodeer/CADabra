from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def _inject_exports(cadquery_code: str, output_dir: Path, input_id: str) -> str:
    step_path = output_dir / f"{input_id}.step"
    stl_path = output_dir / f"{input_id}.stl"

    export_lines = (
        f"result.val().exportStep(r'{step_path}')\n"
        f"cq.Assembly().add(result).save(r'{stl_path}')"
    )

    if "show_object(result)" in cadquery_code:
        return cadquery_code.replace("show_object(result)", export_lines)
    return cadquery_code + "\n" + export_lines


def execute_cadquery(cadquery_code: str, output_dir: str | Path, input_id: str) -> dict[str, str | bool | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = int(os.getenv("CADQUERY_TIMEOUT_SECONDS", "15"))

    final_code = _inject_exports(cadquery_code, output_dir, input_id)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(final_code)
        temp_path = Path(temp_file.name)

    try:
        process = subprocess.run(
            ["python", str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "step_path": None,
            "stl_path": None,
            "error": f"CadQuery execution timed out after {timeout_seconds}s: {exc}",
        }
    finally:
        temp_path.unlink(missing_ok=True)

    if process.returncode != 0:
        stderr = (process.stderr or "").strip()
        stdout = (process.stdout or "").strip()
        return {
            "success": False,
            "step_path": None,
            "stl_path": None,
            "error": stderr or stdout or "CadQuery execution failed",
        }

    step_path = output_dir / f"{input_id}.step"
    stl_path = output_dir / f"{input_id}.stl"
    if not step_path.exists() or step_path.stat().st_size == 0:
        return {
            "success": False,
            "step_path": None,
            "stl_path": None,
            "error": "STEP output missing or empty",
        }
    if not stl_path.exists() or stl_path.stat().st_size == 0:
        return {
            "success": False,
            "step_path": str(step_path),
            "stl_path": None,
            "error": "STL output missing or empty",
        }

    return {
        "success": True,
        "step_path": str(step_path),
        "stl_path": str(stl_path),
        "error": None,
    }
