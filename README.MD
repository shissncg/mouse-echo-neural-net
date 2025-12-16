# Mouse Echo Neural Net
Mouse Echo Neural Net (MENN) is an end-to-end deep learning pipeline for automating preclinical echocardiogram (echo) analysis. This pipeline is designed for the analyses of both b-mode and m-mode mouse echo images/videos. The two major steps in this pipeline are:

- Semantic segmentations (e.g., LV on b-mode)
- Calculations of cardiac parameters, including LV area, volume, ejection fraction, anterior/posterior wall thickness, fractional shortening, and LV Mass

# Forked from 
https://github.com/pfizer-opensource/mouse-echo-neural-net.git

# Installation
First, clone this repository and enter the directory:

## Recommended (uv)
This repo is set up as a `pyproject.toml` project so `uv` can manage the environment and run commands.

    uv sync

Run the auto-detect pipeline (will separate outputs for B-Mode and M-Mode under `output/`):

    uv run menn auto --input-dir path_to_input_dir

This writes an overall `output/run.log` showing every `.dcm` file discovered and whether it was queued (B-Mode/M-Mode) or skipped (unsupported OperatingMode).

Run only b-mode or m-mode:

    uv run menn-bmode --input-dir path_to_input_dir
    uv run menn-mmode --input-dir path_to_input_dir

Launch the UI:

    uv run menn-ui

This starts a Web UI at `http://127.0.0.1:8000` (use `--host/--port` to change).

You can either type an input directory path (recommended for large datasets via volume mounts), or use the upload picker to select a local folder/files in the browser.

## Legacy (conda) - good luck
The original 2017–2021 environment is preserved in `environment.yml`, but it is not recommended for modern systems:

    conda env create -f environment.yml
    conda activate tf-latest
    pip install PyQt5 --upgrade  # optional, for UI

Note that `tf-latest` is the name of the environment. It can be changed directly in `environment.yml`.


# Example result
The following movie demonstrates the automated segmentation results for an sample data set as compared to the manual segmentation (red vs. green).

![SegComp](./images/seg_demo.gif)


# Usage (prediction stage)
To perform the fully automated analysis for b-mode raw DICOM data:

    uv run menn-bmode --input-dir path_to_input_dir

QC outputs (annotated PDF) are written by default; disable with:

    uv run menn-bmode --no-verbose --input-dir path_to_input_dir

Each run also writes `run.log` (per-file status) and `errors.log` (tracebacks) into the output directory.

To perform the fully automated analysis for m-mode raw DICOM data:

    uv run menn-mmode --input-dir path_to_input_dir

QC outputs (annotated PDF) are written by default; disable with:

    uv run menn-mmode --no-verbose --input-dir path_to_input_dir

Each run also writes `run.log` (per-file status) and `errors.log` (tracebacks) into the output directory.

Below is an example figure showing the segmentation results (# of pixels for the LV) for each cardiac phase (x axis) of a b-mode image. The blue curve is the raw result, and the red curve is a smoothed version, which is used to detect the systolic and diastolic phase (green stars and squares):

![CardiacPhases](./images/CardiacPhases.png)

# User Interface
To operate the program from a user interface, use the Web UI:

    uv run menn-web --host 127.0.0.1 --port 8000

In containers, bind to all interfaces and mount your data folder:

    uv run menn-web --host 0.0.0.0 --port 8000 --data-root /data

Then open `http://localhost:8000` and enter the input directory path (under `/data` in the container).

![User Interface](./images/Echo_Segmenter_UI_Window.PNG)

For more detailed information regarding installation, data preparation, and operation of the User Interface see Echo_Segmenter_User_Manual.txt.

# Containerization (Docker)
A minimal `Dockerfile` is included.

## Using a prebuilt image
If your organization publishes a prebuilt MENN container image, you can pull and run it directly instead of building locally.

Pull the image (replace `<IMAGE_NAME>` with the actual registry/image tag you use, for example `registry.example.com/team/menn:latest`):

    docker pull shissncg/menn:latest

Run the container (mount your data directory into `/data`):
- replace </path/to/dicoms> with the absolute path to the dicom files
```
docker run --rm -p 8000:8000 -v </path/to/dicoms>:/data menn
```

## Building locally
If you prefer to build the image yourself, use the included `Dockerfile`:

    docker build -t menn .
    docker run --rm -p 8000:8000 -v /path/to/dicoms:/data menn

# Contribution
Please reach out to chong.duan@pfizer.com, Mary.Montgomery@pfizer.com or dineshh@pfizer.com if you are interested in collaboration and contribution to this project.
