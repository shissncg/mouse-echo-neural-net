from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="menn", description="Mouse Echo Neural Net (MENN) runner.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_auto = sub.add_parser("auto", help="Auto-detect B-Mode/M-Mode DICOMs and run analyses.")
    p_auto.add_argument("--input-dir", required=True, help="Directory containing .dcm files")
    p_auto.add_argument("--output-dir", default="", help="Output directory root (default: <input>/output)")
    p_auto.add_argument("--weights-bmode", default="", help="Override b-mode weights path")
    p_auto.add_argument("--weights-mmode", default="", help="Override m-mode weights path")
    p_auto.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write QC outputs (annotated PDF) to the output directory",
    )
    p_auto.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Exit non-zero if DICOMs with unknown mode are found",
    )

    p_bmode = sub.add_parser("bmode", help="Run b-mode analysis on a directory.")
    p_bmode.add_argument("--input-dir", required=True, help="Directory containing .dcm files")
    p_bmode.add_argument("--output-dir", default="", help="Output directory (default: <input>/output)")
    p_bmode.add_argument("--weights", default="", help="Override b-mode weights path")
    p_bmode.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write QC outputs (annotated PDF) to the output directory",
    )

    p_mmode = sub.add_parser("mmode", help="Run m-mode analysis on a directory.")
    p_mmode.add_argument("--input-dir", required=True, help="Directory containing .dcm files")
    p_mmode.add_argument("--output-dir", default="", help="Output directory (default: <input>/output)")
    p_mmode.add_argument("--weights", default="", help="Override m-mode weights path")
    p_mmode.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write QC outputs (annotated PDF) to the output directory",
    )

    sub.add_parser("ui", help="Launch the Tkinter UI.")

    args = parser.parse_args(argv)

    if args.cmd == "ui":
        from Echo_Segmenter_UI_Full import launch_ui

        launch_ui()
        return 0

    if args.cmd == "bmode":
        from echoanalysis_main import run_bmode_analysis

        run_bmode_analysis(
            input_dir=args.input_dir,
            output_dir=args.output_dir or None,
            verbose=args.verbose,
            weights=args.weights or None,
        )
        return 0

    if args.cmd == "mmode":
        from echoanalysis_mmode_main import run_mmode_analysis

        run_mmode_analysis(
            input_dir=args.input_dir,
            output_dir=args.output_dir or None,
            verbose=args.verbose,
            weights=args.weights or None,
        )
        return 0

    if args.cmd == "auto":
        from echoanalysis_main import run_bmode_analysis
        from echoanalysis_mmode_main import run_mmode_analysis
        from preprocessing_modes import scan_dicoms

        input_dir = Path(args.input_dir).expanduser().resolve()
        out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (input_dir / "output")

        out_root.mkdir(parents=True, exist_ok=True)
        auto_run_log = out_root / "run.log"
        auto_error_log = out_root / "errors.log"
        run_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        auto_run_log.write_text(
            f"run_started_at={run_started_at}\n"
            f"mode=auto\n"
            f"input_dir={input_dir}\n"
            f"output_dir={out_root}\n\n",
            encoding="utf-8",
        )
        auto_error_log.write_text(
            f"run_started_at={run_started_at}\nmode=auto\n\n",
            encoding="utf-8",
        )

        scanned = scan_dicoms(str(input_dir))
        if not scanned:
            raise SystemExit(f"No .dcm files found in {input_dir}")

        bmode_files = []
        mmode_files = []
        skipped = []
        for file_path, mode, err in scanned:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if err:
                skipped.append(file_path)
                with auto_run_log.open("a", encoding="utf-8") as f:
                    f.write(f"{ts}\tSKIP_UNREADABLE\t{file_path}\n")
                with auto_error_log.open("a", encoding="utf-8") as f:
                    f.write(f"file={file_path}\n{err}\n\n")
                continue
            if mode == "B-Mode":
                bmode_files.append(file_path)
                with auto_run_log.open("a", encoding="utf-8") as f:
                    f.write(f"{ts}\tQUEUE_BMODE\t{file_path}\n")
            elif mode == "M-Mode":
                mmode_files.append(file_path)
                with auto_run_log.open("a", encoding="utf-8") as f:
                    f.write(f"{ts}\tQUEUE_MMODE\t{file_path}\n")
            else:
                skipped.append(file_path)
                with auto_run_log.open("a", encoding="utf-8") as f:
                    f.write(f"{ts}\tSKIP_UNSUPPORTED\t{file_path}\tOperatingMode={mode!r}\n")

        if skipped and args.fail_on_unknown:
            raise SystemExit(f"Found {len(skipped)} unsupported/unreadable DICOM(s) in {input_dir}; see {auto_run_log}")

        if not bmode_files and not mmode_files:
            raise SystemExit(f"No B-Mode or M-Mode DICOM files found in {input_dir}; see {auto_run_log}")

        if bmode_files:
            with auto_run_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\tRUN_BMODE\tcount={len(bmode_files)}\n")
            run_bmode_analysis(
                input_dir=input_dir,
                files=bmode_files,
                output_dir=out_root / "B-Mode",
                verbose=args.verbose,
                weights=args.weights_bmode or None,
            )
        if mmode_files:
            with auto_run_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\tRUN_MMODE\tcount={len(mmode_files)}\n")
            run_mmode_analysis(
                input_dir=input_dir,
                files=mmode_files,
                output_dir=out_root / "M-Mode",
                verbose=args.verbose,
                weights=args.weights_mmode or None,
            )

        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with auto_run_log.open("a", encoding="utf-8") as f:
            f.write(
                f"{finished_at}\tSUMMARY\tattempted={len(scanned)}\tqueued_bmode={len(bmode_files)}\tqueued_mmode={len(mmode_files)}\tskipped={len(skipped)}\n"
            )

        return 0

    raise SystemExit(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
