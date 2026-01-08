
# Chest X-Ray Analyzer (Universal Support)

## ⚠️ Usage & Licensing
**This project is licensed under the GNU GPLv3 License.**

This code is provided for **evaluation and hiring assessment purposes only**.
Any use of this codebase, or parts of it, within a proprietary or commercial product without explicit written permission from the author (Shrinivas Nemagoudar) is strictly prohibited and constitutes a violation of the license terms.

© 2026 Shrinivas Nemagoudar. All Rights Reserved.

---

## 🚀 Project Overview
This application is a deep-learning tool designed to analyze Chest X-Ray images. It automatically detects pathologies and segments vital organs.

**Key Features:**
* **Universal File Support:** Works with both **Medical DICOM (.dcm)** files and **Standard Images (PNG/JPG)**.
* **Robust Backend:** Automatically handles complex DICOM compression (like JPEG Lossless) by decoding files on the server.
* **AI Analysis:**
    * **Pathology Detection:** Pneumonia and Pneumothorax (with Heatmaps).
    * **Segmentation:** Lung and Heart visualization.
    * **Predictions:** Patient Age and Sex estimation.

## 📂 Repository Structure
* `backend/` - The Python logic (Gradio, PyTorch, Pydicom).
* `frontend/` - The Web Interface (HTML, CSS, JavaScript).

## 🛠️ How to Run the Project

### Option 1: Run the Frontend Only (Easiest)
The frontend is pre-configured to connect to my live hosted backend on Hugging Face Spaces.

1.  Open the `frontend` folder.
2.  Double-click `index.html` to open it in your browser.
3.  Upload a **.dcm** or **.png** file to test.

### Option 2: Run the Backend Locally
If you want to run the full server on your own machine:

1.  **Install Python 3.10+**
2.  **Install Dependencies:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
3.  **Start the Server:**
    ```bash
    python app.py
    ```
4.  The app will launch locally (usually at `http://127.0.0.1:7860`).

## 🧩 Technical Approach
To solve the issue of reading complex DICOM files, this project uses a **Server-Side Decoding Strategy**.
1.  The frontend sends the raw file to the backend.
2.  The backend attempts to read it as a DICOM dataset using `pydicom`.
3.  If that fails (e.g., it's a PNG), it seamlessly falls back to standard image processing.
4.  The image is normalized to a standard grayscale format before being passed to the AI models.

## 📄 License
Licensed under **GNU GPLv3**. See the `LICENSE` file for full legal details.
