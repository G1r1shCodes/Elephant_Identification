# Building the Elephant Re-ID Executable 🐘

This guide explains how to package the Streamlit application into a standalone Windows Executable (`.exe`) that can be shared with others.

## 1. Prerequisites (For the Builder)

The person building the `.exe` (your friend) needs the following:

1.  **Python 3.10+** installed.
2.  **Git** installed (to clone the repo).
3.  **Visual Studio Build Tools** (C++ compile tools) *may* be required for some packages, but usually pre-built wheels are fine.

## 2. Setup Environment

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/G1r1shCodes/Elephant_ReIdentification.git
    cd Elephant_ReIdentification
    ```

2.  **Create Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Build Tools**:
    ```bash
    pip install pyinstaller shimmy
    ```

## 3. Build the Executable

We have prepared a script to automate the build process.

1.  **Run the Build Script**:
    Double-click `scripts\build_exe.bat` or run it from the terminal:
    ```bash
    scripts\build_exe.bat
    ```

    *This process compares dependencies and bundles them. It may take 5-10 minutes.*

## 4. Post-Build Setup (Crucial Step!)

PyInstaller does **not** bundle large model files by default to keep the build size manageable. You must manually copy them to the output folder.

1.  Go to the new `dist\ElephantID_System\` folder.
2.  **Copy the following files/folders** from your project root into this folder:
    *   `makhna_model.pth` (The trained model)
    *   `gallery_embeddings.pt` (The database of known elephants)
    *   `data/` (The folder containing reference images)

## 5. Running the App

*   Open `dist\ElephantID_System\`
*   Double-click **`ElephantID_System.exe`**

A terminal window will open (this is the backend server), and then the Streamlit app will launch in your default web browser.

## Troubleshooting

*   **"Model not found"**: Ensure you copied `makhna_model.pth` to the same folder as the `.exe`.
*   **"Streamlit not found"**: Ensure you ran `scripts\build_exe.bat` from the environment where `streamlit` is installed.
