import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultText = document.getElementById("resultText");
const labelsContainer = document.getElementById("labelsContainer");
const resultImage = document.getElementById("resultImage");
const heatmapsContainer = document.getElementById("heatmapsContainer");

let selectedFile = null;

// 1. DISABLE BUTTON INITIALLY
analyzeBtn.disabled = true;
resultText.textContent = "Please select a PNG, JPG, or DICOM file to begin.";

// 2. HANDLE FILE SELECTION
fileInput.addEventListener("change", (event) => {
    selectedFile = event.target.files[0];
    
    if (selectedFile) {
        const fileSizeKB = (selectedFile.size / 1024).toFixed(2);
        resultText.textContent = `✅ File Ready: ${selectedFile.name} (${fileSizeKB} KB)\nClick 'Analyze' to upload and process.`;
        resultText.style.color = "#1e3a8a"; 
        resultText.style.fontWeight = "bold";
        
        analyzeBtn.disabled = false;
        analyzeBtn.style.backgroundColor = "#1e3a8a";
        analyzeBtn.style.cursor = "pointer";
    } else {
        analyzeBtn.disabled = true;
        resultText.textContent = "No file selected.";
        selectedFile = null;
    }
});

// 3. HANDLE ANALYZE CLICK
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("No file selected!");
        return;
    }

    const DETECTION_THRESHOLD = 0.10; 

    // UI Feedback
    resultText.textContent = "⏳ Uploading file to server... Please wait.";
    resultText.style.color = "#333";
    labelsContainer.innerHTML = "";
    heatmapsContainer.innerHTML = ""; 
    resultImage.style.display = "none";
    resultImage.src = "";
    
    analyzeBtn.disabled = true; 
    analyzeBtn.textContent = "Processing...";
    analyzeBtn.style.backgroundColor = "#94a3b8"; 

    try {
        // --- CONNECT TO YOUR NEW FINAL SPACE ---
        const client = await Client.connect("shrinusn77/chest-xray-analyzer-final");
        
        // --- SEND RAW FILE ---
        // The backend 'load_image_safe' will figure out if it's PNG or DICOM
        const result = await client.predict("/predict", {
            Radiograph_path: selectedFile, 
        });

        // --- DISPLAY RESULTS ---
        resultText.textContent = result.data[0];

        const labelsData = result.data[1];
        if (labelsData && labelsData.confidences) {
            labelsData.confidences.forEach(item => {
                const p = document.createElement("p");
                p.className = "label-row";
                p.innerHTML = `
                    <span>${item.label}</span>
                    <span>${(item.confidence * 100).toFixed(0)}%</span>
                `;
                labelsContainer.appendChild(p);

                if ((item.label === 'Pneumonia' || item.label === 'Pneumothorax') && item.confidence > DETECTION_THRESHOLD) {
                    const heatmapIndex = item.label === 'Pneumonia' ? 3 : 4;
                    const heatmapImage = result.data[heatmapIndex]; 
                    
                    if (heatmapImage && heatmapImage.url) {
                        const wrapper = document.createElement('div');
                        wrapper.className = 'heatmap-wrapper';
                        wrapper.innerHTML = `<h4>${item.label} Heatmap:</h4><img src="${heatmapImage.url}" />`;
                        heatmapsContainer.appendChild(wrapper);
                    }
                }
            });
        }

        const finalImage = result.data[2];
        if (finalImage && finalImage.url) {
            resultImage.src = finalImage.url;
            resultImage.style.display = "block";
        }

    } catch (err) {
        console.error("Error during analysis:", err);
        resultText.textContent = "❌ Error: " + err.message;
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze X-ray";
        analyzeBtn.style.backgroundColor = "#1e3a8a";
    }
});