# -*- coding: utf-8 -*-
"""
Main function for b-mode echo segmenter
Created on Fri Jun 26 10:49:25 2020

@author: DUANC01
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import traceback
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from skimage import io
import scipy
import pydicom
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import cv2

from util_bmode import dicom_preprocess, computeMetrics, findcardiacpeaks, getRes_rawDICOM, getRes, postprocess_masks

def default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "model_weights" / "weights_ECHO_clean.h5"


def _normalize_frames(img_raw: np.ndarray) -> np.ndarray:
    if img_raw.ndim == 4:
        return img_raw
    if img_raw.ndim == 3:
        if img_raw.shape[-1] in (1, 3, 4):
            return img_raw[None, ...]
        return img_raw
    if img_raw.ndim == 2:
        return img_raw[None, ...]
    raise ValueError(f"Unsupported DICOM pixel array shape: {img_raw.shape}")


def run_bmode_analysis(
    *,
    input_dir: Optional[str | Path] = None,
    files: Optional[Sequence[str | Path]] = None,
    output_dir: Optional[str | Path] = None,
    verbose: bool = False,
    weights: Optional[str | Path] = None,
) -> Path:
    if input_dir is None and not files:
        raise ValueError("Provide either input_dir or files")

    if files:
        file_paths = [Path(p).expanduser().resolve() for p in files]
        base_dir = Path(input_dir).expanduser().resolve() if input_dir else file_paths[0].parent
    else:
        base_dir = Path(input_dir).expanduser().resolve()  # type: ignore[arg-type]
        file_paths = sorted(
            p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"
        )

    if not file_paths:
        raise FileNotFoundError(f"No .dcm files found in {base_dir}")

    weights_path = Path(weights).expanduser().resolve() if weights else default_weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else (base_dir / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = out_dir / "run.log"
    error_log_path = out_dir / "errors.log"
    run_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_log_path.write_text(
        f"run_started_at={run_started_at}\n"
        f"mode=bmode\n"
        f"input_dir={base_dir}\n"
        f"output_dir={out_dir}\n"
        f"weights={weights_path}\n",
        encoding="utf-8",
    )
    error_log_path.write_text(
        f"run_started_at={run_started_at}\nmode=bmode\n\n",
        encoding="utf-8",
    )

    from util_nn_bmode import get_unet

    model = get_unet()
    model.load_weights(str(weights_path))

    sys_areas: list[float] = []
    dia_areas: list[float] = []
    sys_volumes: list[float] = []
    dia_volumes: list[float] = []
    ejection_fractions: list[float] = []

    sys_segs: list[np.ndarray] = []
    dia_segs: list[np.ndarray] = []
    animal_ids: list[str] = []
    attempted = 0
    failed = 0

    for filepath in file_paths:
        attempted += 1
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with run_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{started_at}\tSTART\t{filepath}\n")
        try:
            data = pydicom.dcmread(str(filepath))
            img_raw = _normalize_frames(data.pixel_array)

            try:
                res_x, res_y = getRes_rawDICOM(data)
            except Exception:
                res_x, res_y = getRes(data)
                res_x, res_y = res_x * 10, res_y * 10

            images = []
            org_x = org_y = None
            for i in range(img_raw.shape[0]):
                image = dicom_preprocess(img_raw[i])
                org_y, org_x = image.shape
                image = np.uint8(np.round(resize(np.float32(image), (256, 256), preserve_range=True)))
                images.append(image)

            if org_x is None or org_y is None:
                raise RuntimeError("Failed to extract frames")

            imgs = np.stack(images)[..., np.newaxis]
            res_x, res_y = res_x * org_x / 256, res_y * org_y / 256

            mean = 92
            std = 57
            imgs = imgs.astype(np.float32)
            imgs = (imgs - mean) / std

            imgs_mask_test = model.predict(imgs, verbose=1, batch_size=1)
            imgs_mask_test = (imgs_mask_test >= 0.5).astype(np.uint8)
            imgs_mask_test = postprocess_masks(imgs_mask_test, contappr=False)

            systoles, diastoles = findcardiacpeaks(imgs_mask_test)
            sys_area, dia_area, sys_volume, dia_volume, ef = computeMetrics(
                imgs_mask_test, systoles, diastoles, res_x, res_y
            )

            animal_ids.append(filepath.stem)
            sys_areas.append(sys_area)
            dia_areas.append(dia_area)
            sys_volumes.append(sys_volume)
            dia_volumes.append(dia_volume)
            ejection_fractions.append(ef)

            if verbose:
                sys_idx = systoles[0]
                dia_idx = diastoles[0]

                a = rescale_intensity(imgs[sys_idx, :, :, 0], out_range=(-1, 1))
                b = imgs_mask_test[sys_idx][:, :, 0].astype("uint8")
                ab = mark_boundaries(a, b)
                sys_segs.append(rescale_intensity(ab, out_range=(0, 255)).astype("uint8"))

                a = rescale_intensity(imgs[dia_idx, :, :, 0], out_range=(-1, 1))
                b = imgs_mask_test[dia_idx][:, :, 0].astype("uint8")
                ab = mark_boundaries(a, b)
                dia_segs.append(rescale_intensity(ab, out_range=(0, 255)).astype("uint8"))

            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{started_at}\tOK\t{filepath}\n")
        except Exception as e:
            failed += 1
            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{started_at}\tERROR\t{filepath}\t{type(e).__name__}: {e}\n")
            with error_log_path.open("a", encoding="utf-8") as f:
                f.write(f"file={filepath}\n{traceback.format_exc()}\n\n")
            continue

    if not animal_ids:
        raise RuntimeError(f"All files failed; see {error_log_path}")

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with run_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{finished_at}\tSUMMARY\tattempted={attempted}\tok={len(animal_ids)}\terrors={failed}\n"
        )

    df = pd.DataFrame(
        data={
            "animal_id": animal_ids,
            "sys_area (mm2)": sys_areas,
            "dia_area (mm2)": dia_areas,
            "sys_volume (mm3)": sys_volumes,
            "dia_volume (mm3)": dia_volumes,
            "ejection_fraction": ejection_fractions,
        }
    )
    csv_path = out_dir / "output.csv"
    df.to_csv(csv_path, index=False)

    if verbose:
        with PdfPages(out_dir / "output_images.pdf") as pdf:
            for i in range(len(animal_ids)):
                plt.figure(figsize=(8, 6))
                plt.suptitle(f"{animal_ids[i]}", fontsize=14)
                plt.subplot(121)
                plt.imshow(sys_segs[i])
                plt.title("systole")
                plt.subplot(122)
                plt.imshow(dia_segs[i])
                plt.title("diastole")
                pdf.savefig()
                plt.close()

            d = pdf.infodict()
            d["Title"] = "Segmentation Results"
            d["Author"] = "Chong Duan"
            d["Subject"] = "Systolic and diastolic segmentations"
            d["Keywords"] = "Automated Echocardiography Analysis"
            d["CreationDate"] = datetime.datetime(2020, 9, 3)
            d["ModDate"] = datetime.datetime.today()

    return csv_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run MENN b-mode analysis on DICOM files.")
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", default="", help="Path to input directory")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="", help="Path to output directory")
    parser.add_argument("--weights", default="", help="Path to b-mode model weights (.h5)")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write QC outputs (annotated PDF) to the output directory",
    )
    args = parser.parse_args(argv)

    if not args.input_dir:
        parser.error("No input directory provided, add --input_dir/--input-dir")

    run_bmode_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir or None,
        verbose=args.verbose,
        weights=args.weights or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
