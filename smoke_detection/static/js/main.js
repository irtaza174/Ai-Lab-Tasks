// Smoke Detection System - Main JavaScript

// Global variables
let cameraStream = null;
let isProcessing = false;
let detectionInterval = null;
let alertHistory = [];
let currentThreshold = 0.75;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const previewImage = document.getElementById('previewImage');
const uploadResult = document.getElementById('uploadResult');
const cameraFeed = document.getElementById('cameraFeed');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const cameraStatusText = document.getElementById('cameraStatusText');
const confidenceText = document.getElementById('confidenceText');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const alertModal = document.getElementById('alertModal');
const closeAlertBtn = document.getElementById('closeAlertBtn');
const alertTime = document.getElementById('alertTime');
const alertConfidence = document.getElementById('alertConfidence');
const alertHistoryContainer = document.getElementById('alertHistory');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const loadingOverlay = document.getElementById('loadingOverlay');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadAlertHistory();
});

// Setup Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Upload button
    uploadBtn.addEventListener('click', uploadImage);
    
    // Camera controls
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
    
    // Threshold slider
    thresholdSlider.addEventListener('input', updateThreshold);
    
    // Alert modal
    closeAlertBtn.addEventListener('click', closeAlert);
    
    // Clear history
    clearHistoryBtn.addEventListener('click', clearHistory);
}

// File Upload Handlers
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        displayPreview(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        fileInput.files = event.dataTransfer.files;
        displayPreview(file);
    }
}

function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
        uploadBtn.disabled = false;
    };
    
    reader.readAsDataURL(file);
}

// Upload Image
async function uploadImage() {
    const file = fileInput.files[0];
    if (!file) return;
    
    showLoading();
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayUploadResult(data);
            
            if (data.smoke_detected) {
                showAlert(data.timestamp, data.confidence);
                addToHistory(data.timestamp, data.confidence);
            }
        } else {
            showError(data.error || 'Upload failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displayUploadResult(data) {
    const resultClass = data.smoke_detected ? 'smoke-detected' : 'no-smoke';
    const resultIcon = data.smoke_detected ? '🚨' : '✅';
    const resultText = data.smoke_detected ? 'SMOKE DETECTED!' : 'No Smoke Detected';
    
    uploadResult.className = `result-container ${resultClass}`;
    uploadResult.innerHTML = `
        <div class="result-header">
            ${resultIcon} ${resultText}
        </div>
        <div class="result-details">
            <div class="result-item">
                <span class="result-label">Confidence</span>
                <span class="result-value">${(data.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Timestamp</span>
                <span class="result-value">${data.timestamp}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Threshold</span>
                <span class="result-value">${(data.threshold * 100).toFixed(0)}%</span>
            </div>
        </div>
    `;
    uploadResult.style.display = 'block';
}

// Camera Functions
async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        
        cameraFeed.srcObject = cameraStream;
        cameraFeed.classList.add('active');
        cameraPlaceholder.style.display = 'none';
        
        startCameraBtn.style.display = 'none';
        stopCameraBtn.style.display = 'inline-block';
        
        cameraStatusText.textContent = 'Running';
        cameraStatusText.style.color = '#10b981';
        
        // Start detection loop
        startDetection();
        
    } catch (error) {
        showError('Camera access denied: ' + error.message);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    cameraFeed.classList.remove('active');
    cameraPlaceholder.style.display = 'flex';
    
    startCameraBtn.style.display = 'inline-block';
    stopCameraBtn.style.display = 'none';
    
    cameraStatusText.textContent = 'Stopped';
    cameraStatusText.style.color = '#6b7280';
    confidenceText.textContent = '--';
    
    // Stop detection loop
    stopDetection();
}

function startDetection() {
    // Process frame every 1 second
    detectionInterval = setInterval(async () => {
        if (!isProcessing && cameraStream) {
            await processFrame();
        }
    }, 1000);
}

function stopDetection() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
}

async function processFrame() {
    isProcessing = true;
    
    try {
        // Capture frame from video
        const canvas = cameraCanvas;
        const context = canvas.getContext('2d');
        
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        
        context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server
        const response = await fetch('/predict_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update confidence display
            const confidencePercent = (data.confidence * 100).toFixed(1);
            confidenceText.textContent = `${confidencePercent}%`;
            confidenceText.style.color = data.smoke_detected ? '#ef4444' : '#10b981';
            
            // Show alert if smoke detected
            if (data.smoke_detected) {
                showAlert(data.timestamp, data.confidence);
                addToHistory(data.timestamp, data.confidence);
                
                // Optional: Stop camera on detection
                // stopCamera();
            }
        }
    } catch (error) {
        console.error('Frame processing error:', error);
    } finally {
        isProcessing = false;
    }
}

// Alert Functions
function showAlert(timestamp, confidence) {
    alertTime.textContent = timestamp;
    alertConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
    
    alertModal.classList.add('show');
    
    // Play alert sound (optional)
    playAlertSound();
}

function closeAlert() {
    alertModal.classList.remove('show');
}

function playAlertSound() {
    // Create a simple beep sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
}

// Alert History
function addToHistory(timestamp, confidence) {
    const alert = {
        timestamp: timestamp,
        confidence: confidence,
        id: Date.now()
    };
    
    alertHistory.unshift(alert);
    
    // Keep only last 10 alerts
    if (alertHistory.length > 10) {
        alertHistory = alertHistory.slice(0, 10);
    }
    
    saveAlertHistory();
    renderAlertHistory();
}

function renderAlertHistory() {
    if (alertHistory.length === 0) {
        alertHistoryContainer.innerHTML = '<p class="no-alerts">No alerts yet</p>';
        return;
    }
    
    alertHistoryContainer.innerHTML = alertHistory.map(alert => `
        <div class="alert-item">
            <div class="alert-item-header">
                <span class="alert-item-time">🚨 ${alert.timestamp}</span>
                <span class="alert-item-confidence">Confidence: ${(alert.confidence * 100).toFixed(1)}%</span>
            </div>
        </div>
    `).join('');
}

function clearHistory() {
    if (confirm('Clear all alert history?')) {
        alertHistory = [];
        saveAlertHistory();
        renderAlertHistory();
    }
}

function saveAlertHistory() {
    localStorage.setItem('smokeAlertHistory', JSON.stringify(alertHistory));
}

function loadAlertHistory() {
    const saved = localStorage.getItem('smokeAlertHistory');
    if (saved) {
        alertHistory = JSON.parse(saved);
        renderAlertHistory();
    }
}

// Threshold Control
function updateThreshold() {
    const value = thresholdSlider.value / 100;
    currentThreshold = value;
    thresholdValue.textContent = value.toFixed(2);
    
    // Update server threshold
    fetch('/set_threshold', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ threshold: value })
    }).catch(error => {
        console.error('Failed to update threshold:', error);
    });
}

// Utility Functions
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showError(message) {
    alert('Error: ' + message);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (cameraStream) {
        stopCamera();
    }
});
