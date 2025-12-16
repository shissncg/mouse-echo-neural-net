# -*- coding: utf-8 -*-
"""
Created on 2/3/21
@author: MONTGM11

Echo_Segmenter_UI_Full.py
User interface for automated echocardiography tool. Combines b-mode and m-mode tools
"""

from __future__ import annotations

# Import libraries
from pathlib import Path
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox as msg

def launch_ui() -> None:
    # Create GUI
    window = tk.Tk()
    window.title("Echocardiography Segmenter")
    window.geometry("900x200")

    appTitle = tk.Label(
        window, text="Echocardiography Segmenter", font=("calibri", 20, "bold"), fg="forest green"
    )
    appTitle.grid(column=3, row=0)

    # ----------------------- Formatting -----------------------
    offsetBlank = tk.Label(text="", width=4, height=2)
    offsetBlank.grid(column=0, row=0)
    for r in (2, 4, 6, 8, 12):
        tk.Label(text="", width=1, height=1).grid(column=0, row=r)

    # --------------------- Button to select study directory ---------------------
    studyDir = tk.StringVar(value="None Selected")

    def button1clicked() -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            studyDir.set(chosen)

    tk.Label(window, text="Input Directory:", justify="right", width=20).grid(column=1, row=1)
    tk.Label(
        window, textvariable=studyDir, font=("arial", 8), justify="center", bg="white", width=90
    ).grid(column=2, row=1, columnspan=3)
    tk.Button(window, text="Select", command=button1clicked, justify="right", width=10).grid(column=5, row=1)

    # ------------------------ Verbose Check Box ------------------------------
    verbose = tk.BooleanVar(value=True)
    ttk.Checkbutton(window, text="Save QC Results", variable=verbose).grid(column=3, row=3)

    # ----------------------- Run button ----------------------------
    def runButtonClicked() -> None:
        input_dir = studyDir.get()
        if not input_dir or input_dir == "None Selected":
            msg.showinfo(message="No input directory selected")
            return

        from preprocessing_modes import classify_dicoms
        from echoanalysis_main import run_bmode_analysis
        from echoanalysis_mmode_main import run_mmode_analysis

        try:
            classified = classify_dicoms(input_dir)
            bmode_files = classified["B-Mode"]
            mmode_files = classified["M-Mode"]
            unknown_files = classified["Unknown"]

            if not bmode_files and not mmode_files:
                msg.showinfo(message="No B-Mode or M-Mode DICOM files found in the selected directory.")
                return

            out_root = Path(input_dir) / "output"
            if bmode_files:
                run_bmode_analysis(
                    input_dir=input_dir,
                    files=bmode_files,
                    output_dir=out_root / "B-Mode",
                    verbose=bool(verbose.get()),
                )
            if mmode_files:
                run_mmode_analysis(
                    input_dir=input_dir,
                    files=mmode_files,
                    output_dir=out_root / "M-Mode",
                    verbose=bool(verbose.get()),
                )

            if unknown_files:
                msg.showinfo(
                    message=f"Analysis complete.\n\nIgnored {len(unknown_files)} DICOM(s) with unknown mode."
                )
            else:
                msg.showinfo(message="Analysis complete!")
        except Exception as e:
            msg.showinfo(message=f"There was an issue with this run:\n\n{e}")

    tk.Button(window, text="Run", font=("Arial", 18), command=runButtonClicked).grid(column=3, row=5)
    window.mainloop()


if __name__ == "__main__":
    launch_ui()
