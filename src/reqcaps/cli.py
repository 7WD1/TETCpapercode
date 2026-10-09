"""CLI entry point. By default validate/build/query quietly, without result files."""
import argparse
import importlib
import json
import sys
from pathlib import Path
import numpy as np
from .config import Config
from .pipeline import ReQCAPS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and query a local ReQ-CAPS artifact")
    parser.add_argument("--model", required=True, help="Python module:attribute exposing predict(X) or callable(X)")
    parser.add_argument("--factory", action="store_true", help="call the model attribute with no arguments to obtain a predictor")
    parser.add_argument("--background", required=True, type=Path, help="processed numeric CSV, no header")
    parser.add_argument("--instances", required=True, type=Path, help="processed numeric CSV, one audited instance per row")
    parser.add_argument("--config", type=Path, help="JSON fields matching reqcaps.Config")
    parser.add_argument("--output", type=Path, help="optional JSON path; no output is generated when omitted")
    args = parser.parse_args(argv)
    try:
        fields = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        config = Config(**fields)
        module_name, attribute = args.model.split(":", 1)
        model = getattr(importlib.import_module(module_name), attribute)
        if args.factory:
            model = model()
        background = np.loadtxt(args.background, delimiter=",", ndmin=2)
        instances = np.loadtxt(args.instances, delimiter=",", ndmin=2)
        engine = ReQCAPS(config)
        artifacts = engine.build_batch(model, instances, background)
        for artifact in artifacts:
            artifact.answer_all()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps([item.to_dict() for item in artifacts],
                                             indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return 0
    except (ValueError, TypeError, RuntimeError, ImportError, AttributeError, OSError) as error:
        print(f"reqcaps: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

