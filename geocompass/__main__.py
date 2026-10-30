"""Command line entry point: ``python -m geocompass``."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(
            "GeoCompass: calibrated skill selection and tiered memory\n\n"
            "Commands:\n"
            "  experiment [--seeds 0,1,2] [--k 8] [--budget 3] [--output DIR]\n"
            "             [--retriever hashed|bm25|hybrid] [--no-ablations]\n"
            "  report     [--results PATH] [--output DIR]\n"
        )
        return 0
    command, rest = argv[0], argv[1:]
    if command == "experiment":
        from .experiment import main as run_main
        return run_main(rest)
    if command == "report":
        sys.argv = ["geocompass-report", *rest]
        from .report import main as report_main
        return report_main()
    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
