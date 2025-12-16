# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:21:00 2021

@author: MONTGM11

Main function for m-mode analysis adapted from DUANC01's echoanalysis_main for
b-mode analysis
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import traceback
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
import pydicom
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from util_mmode import getRes_rawDICOM, compute_MMode_metrics, postprocess, getRes, crop_frame

def default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "model_weights" / "weights_MMode_clean_v4.h5"


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


def run_mmode_analysis(
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
        f"mode=mmode\n"
        f"input_dir={base_dir}\n"
        f"output_dir={out_dir}\n"
        f"weights={weights_path}\n",
        encoding="utf-8",
    )
    error_log_path.write_text(
        f"run_started_at={run_started_at}\nmode=mmode\n\n",
        encoding="utf-8",
    )

    from util_nn_mmode import get_unet

    model = get_unet()
    model.load_weights(str(weights_path))

    LVAWs_s: list[float] = []
    LVAWs_d: list[float] = []
    LVIDs_s: list[float] = []
    LVIDs_d: list[float] = []
    LVPWs_s: list[float] = []
    LVPWs_d: list[float] = []
    FSs: list[float] = []
    LV_Masses: list[float] = []
    LV_Mass_Cors: list[float] = []
    Heart_Rates: list[float] = []
    frames_used: list[int] = []
    segs = []
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
                res_x, res_y = res_x * 1000, res_y * 10

            images = []
            org_x = org_y = None
            for z in range(img_raw.shape[0]):
                frame = img_raw[z]
                image = crop_frame(np.squeeze(frame))
                org_y, org_x = image.shape
                image = rescale_intensity(np.float32(image), out_range=(0, 255))
                image = np.uint8(np.round(resize(image, (256, 256), preserve_range=True)))
                images.append(image)

            if org_x is None or org_y is None:
                raise RuntimeError("Failed to extract frames")

            imgs = np.stack(images)[..., np.newaxis]
            res_x, res_y = res_x * org_x / 256, res_y * org_y / 256

            mean = 127
            std = 51
            imgs = imgs.astype(np.float32)
            imgs = (imgs - mean) / std

            imgs_mask_test = model.predict(imgs, verbose=1, batch_size=1)
            conf = np.mean(np.abs(imgs_mask_test - 0.5), axis=(1, 2, 3)).tolist()

            imgs_mask_test = np.round(imgs_mask_test).astype(np.uint8)
            label = np.squeeze(postprocess(imgs_mask_test))

            cutoff = int(np.floor(256 * 0.1))
            label = label[:, :, cutoff - 1 : -cutoff]
            imgs = imgs[:, :, cutoff - 1 : -cutoff, :]

            LVAW_s_tmp = []
            LVAW_d_tmp = []
            LVID_s_tmp = []
            LVID_d_tmp = []
            LVPW_s_tmp = []
            LVPW_d_tmp = []
            FS_tmp = []
            LV_Mass_tmp = []
            LV_Mass_Cor_tmp = []
            Heart_Rate_tmp = []
            for z in range(label.shape[0]):
                metrics = compute_MMode_metrics(np.squeeze(label[z, :, :]), res_y, res_x, agg_fn=np.median)
                LVAW_s, LVAW_d, LVID_s, LVID_d, LVPW_s, LVPW_d, FS, LV_Mass, LV_Mass_Cor, Heart_Rate = metrics
                LVAW_s_tmp.append(LVAW_s)
                LVAW_d_tmp.append(LVAW_d)
                LVID_s_tmp.append(LVID_s)
                LVID_d_tmp.append(LVID_d)
                LVPW_s_tmp.append(LVPW_s)
                LVPW_d_tmp.append(LVPW_d)
                FS_tmp.append(FS)
                LV_Mass_tmp.append(LV_Mass)
                LV_Mass_Cor_tmp.append(LV_Mass_Cor)
                Heart_Rate_tmp.append(Heart_Rate)

            idx = int(np.argmax(conf))

            animal_ids.append(filepath.stem)
            frames_used.append(idx)
            LVAWs_s.append(LVAW_s_tmp[idx])
            LVAWs_d.append(LVAW_d_tmp[idx])
            LVIDs_s.append(LVID_s_tmp[idx])
            LVIDs_d.append(LVID_d_tmp[idx])
            LVPWs_s.append(LVPW_s_tmp[idx])
            LVPWs_d.append(LVPW_d_tmp[idx])
            FSs.append(FS_tmp[idx])
            LV_Masses.append(LV_Mass_tmp[idx])
            LV_Mass_Cors.append(LV_Mass_Cor_tmp[idx])
            Heart_Rates.append(Heart_Rate_tmp[idx])

            if verbose:
                a = rescale_intensity(imgs[idx, :, :, 0], out_range=(-1, 1))
                b = label[idx, :, :]
                ab = mark_boundaries(a, b)
                segs.append(rescale_intensity(ab, out_range=(0, 255)).astype("uint8"))

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
            "LVAW_sys (mm)": LVAWs_s,
            "LVAW_dia (mm)": LVAWs_d,
            "LVPW_sys (mm)": LVPWs_s,
            "LVPW_dia (mm)": LVPWs_d,
            "LVID_sys (mm)": LVIDs_s,
            "LVID_dia (mm)": LVIDs_d,
            "FS (%)": FSs,
            "LV_Mass (mg)": LV_Masses,
            "LV_Mass_Cor (mg)": LV_Mass_Cors,
            "Heart_Rate": Heart_Rates,
            "frame_used": frames_used,
        }
    )
    csv_path = out_dir / "output.csv"
    df.to_csv(csv_path, index=False)

    if verbose:
        with PdfPages(out_dir / "output_images.pdf") as pdf:
            for i in range(len(segs)):
                plt.figure(figsize=(8, 6))
                plt.title(animal_ids[i], fontsize=14)
                plt.imshow(segs[i])
                pdf.savefig()
                plt.close()

            d = pdf.infodict()
            d["Title"] = "Segmentation Results"
            d["Author"] = "Chong Duan"
            d["Subject"] = "M-mode segmentations"
            d["Keywords"] = "Automated Echocardiography Analysis"
            d["CreationDate"] = datetime.datetime(2021, 1, 21)
            d["ModDate"] = datetime.datetime.today()

    return csv_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run MENN m-mode analysis on DICOM files.")
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", default="", help="Path to input directory")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="", help="Path to output directory")
    parser.add_argument("--weights", default="", help="Path to m-mode model weights (.h5)")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write QC outputs (annotated PDF) to the output directory",
    )
    args = parser.parse_args(argv)

    if not args.input_dir:
        parser.error("No input directory provided, add --input_dir/--input-dir")

    run_mmode_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir or None,
        verbose=args.verbose,
        weights=args.weights or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
