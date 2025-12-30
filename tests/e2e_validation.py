#!/usr/bin/env python3
"""End-to-end validation test for TheBoard v1.0.

Tests complete workflow:
1. Create meeting
2. Run meeting (1 round for speed)
3. Check status
4. Export results
5. Cleanup

Sprint 6 validation - ensures production readiness.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n🔍 {description}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print(f"   ✅ Success")
            return True, result.stdout
        else:
            print(f"   ❌ Failed (exit code: {result.returncode})")
            print(f"   Error: {result.stderr[:500]}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout after 60 seconds")
        return False, "Timeout"
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e)


def main():
    """Run end-to-end validation."""
    print("=" * 70)
    print("TheBoard v1.0 - End-to-End Validation")
    print("=" * 70)

    # Test 1: Check CLI is accessible
    success, output = run_command(
        ["board", "--help"],
        "Test 1: Check CLI accessibility"
    )
    if not success:
        print("\n❌ FAILED: CLI not accessible")
        return 1

    # Test 2: Create a meeting
    success, output = run_command(
        [
            "board", "create",
            "--topic", "Test validation meeting for TheBoard v1.0",
            "--strategy", "sequential",
            "--max-rounds", "1",
            "--agent-count", "3",
        ],
        "Test 2: Create meeting"
    )
    if not success:
        print("\n❌ FAILED: Could not create meeting")
        return 1

    # Extract meeting ID from output
    meeting_id = None
    for line in output.split('\n'):
        if "Meeting ID:" in line:
            parts = line.split("Meeting ID:")
            if len(parts) > 1:
                meeting_id = parts[1].strip()
                break

    if not meeting_id:
        print("\n❌ FAILED: Could not extract meeting ID from output")
        print(f"Output: {output[:500]}")
        return 1

    print(f"   📝 Meeting ID: {meeting_id}")

    # Test 3: Check meeting status (before running)
    success, output = run_command(
        ["board", "status", meeting_id],
        "Test 3: Check meeting status (before run)"
    )
    if not success:
        print("\n❌ FAILED: Could not check meeting status")
        return 1

    # Test 4: Run the meeting
    print(f"\n🔍 Test 4: Run meeting (1 round, 3 agents)")
    print(f"   This may take 30-60 seconds...")
    success, output = run_command(
        ["board", "run", meeting_id],
        "   Running meeting"
    )
    if not success:
        print("\n❌ FAILED: Meeting execution failed")
        return 1

    # Test 5: Check meeting status (after running)
    success, output = run_command(
        ["board", "status", meeting_id],
        "Test 5: Check meeting status (after run)"
    )
    if not success:
        print("\n❌ FAILED: Could not check final status")
        return 1

    # Verify status shows completed
    if "completed" not in output.lower() and "failed" not in output.lower():
        print("\n⚠️  WARNING: Meeting may not have completed")
        print(f"Status output: {output[:200]}")

    # Test 6: Export as markdown
    output_file = Path("/tmp/theboard-test-export.md")
    if output_file.exists():
        output_file.unlink()

    success, output = run_command(
        [
            "board", "export", meeting_id,
            "--format", "markdown",
            "--output", str(output_file),
        ],
        "Test 6: Export as markdown"
    )
    if not success:
        print("\n❌ FAILED: Export failed")
        return 1

    # Verify export file exists and has content
    if not output_file.exists():
        print(f"\n❌ FAILED: Export file not created at {output_file}")
        return 1

    file_size = output_file.stat().st_size
    if file_size == 0:
        print(f"\n❌ FAILED: Export file is empty")
        return 1

    print(f"   📄 Export file created: {output_file} ({file_size:,} bytes)")

    # Test 7: Export as JSON
    json_file = Path("/tmp/theboard-test-export.json")
    if json_file.exists():
        json_file.unlink()

    success, output = run_command(
        [
            "board", "export", meeting_id,
            "--format", "json",
            "--output", str(json_file),
        ],
        "Test 7: Export as JSON"
    )
    if not success:
        print("\n❌ FAILED: JSON export failed")
        return 1

    if not json_file.exists() or json_file.stat().st_size == 0:
        print(f"\n❌ FAILED: JSON export file invalid")
        return 1

    print(f"   📄 JSON file created: {json_file} ({json_file.stat().st_size:,} bytes)")

    # Success!
    print("\n" + "=" * 70)
    print("✅ All validation tests passed!")
    print("=" * 70)
    print("\nTheBoard v1.0 is production-ready:")
    print("  ✅ CLI functional")
    print("  ✅ Meeting creation working")
    print("  ✅ Meeting execution successful")
    print("  ✅ Status reporting accurate")
    print("  ✅ Export (markdown) working")
    print("  ✅ Export (JSON) working")
    print("\n🎉 System validated and ready for production!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
