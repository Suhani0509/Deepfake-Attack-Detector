// Frontend JavaScript for Deepfake Detection App
class DeepfakeDetectorApp {
    constructor() {
        this.initializeApp();
        this.bindEvents();
    }

    initializeApp() {
        this.elements = {
            uploadForm: document.getElementById('uploadForm'),
            videoInput: document.getElementById('videoInput'),
            analyzeBtn: document.getElementById('analyzeBtn'),
            btnText: document.querySelector('.btn-text'),
            btnLoading: document.getElementById('btnLoading'),
            resultsSection: document.getElementById('resultsSection'),
            resultIcon: document.getElementById('resultIcon'),
            resultText: document.getElementById('resultText'),
            confidenceFill: document.getElementById('confidenceFill'),
            confidenceValue: document.getElementById('confidenceValue'),
            explanationText: document.getElementById('explanationText'),
            resultsMeta: document.getElementById('resultsMeta'),
            filePreview: document.getElementById('filePreview')
        };

        this.setInitialState();
    }

    setInitialState() {
        this.elements.resultsSection.style.display = 'none';
        this.updateResultsUI('ready', 'Ready to Analyze', 0, 
            'Upload a video file to begin analysis. Our AI will examine the video for deepfake indicators.');
    }

    bindEvents() {
        this.elements.uploadForm.addEventListener('submit', (e) => this.handleFormSubmit(e));
        this.elements.videoInput.addEventListener('change', (e) => this.handleFileSelect(e));
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.displayFilePreview(file);
        }
    }

    displayFilePreview(file) {
        const fileSize = this.formatFileSize(file.size);
        this.elements.filePreview.innerHTML = `
            <div class="file-preview-content">
                <i class="fas fa-file-video"></i>
                <div class="file-info">
                    <strong>${file.name}</strong>
                    <span>${fileSize}</span>
                </div>
            </div>
        `;
        this.elements.filePreview.classList.add('active');
    }

    async handleFormSubmit(event) {
        event.preventDefault();
        
        const file = this.elements.videoInput.files[0];
        if (!file) {
            this.showError('Please select a video file');
            return;
        }

        // Validate file size (100MB max)
        if (file.size > 100 * 1024 * 1024) {
            this.showError('File size must be less than 100MB');
            return;
        }

        this.setLoadingState(true);
        this.showResultsSection();
        this.updateResultsUI('analyzing', 'Analyzing Video...', 0, 
            'Processing your video. This may take a few seconds...');

        try {
            const formData = new FormData();
            formData.append('video', file);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.displayResults(data);
            } else {
                this.showError(data.error || 'An error occurred during analysis');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.setLoadingState(false);
        }
    }

    displayResults(data) {
        const isReal = data.result === 'REAL';
        const status = isReal ? 'real' : 'fake';
        
        this.updateResultsUI(
            status,
            isReal ? 'REAL VIDEO' : 'DEEPFAKE DETECTED',
            data.confidence,
            data.explanation
        );

        // Update metadata
        this.elements.resultsMeta.textContent = `Analysis completed in ${data.analysis_time}s`;
    }

    updateResultsUI(status, text, confidence, explanation) {
        // Update icon and text
        const icon = this.elements.resultIcon;
        icon.className = 'result-icon';
        icon.innerHTML = this.getStatusIcon(status);
        
        const resultText = this.elements.resultText;
        resultText.textContent = text;
        resultText.className = `result-text ${status}`;

        // Update confidence meter
        this.animateConfidenceMeter(confidence);

        // Update explanation
        this.elements.explanationText.textContent = explanation;
    }

    getStatusIcon(status) {
        const icons = {
            ready: '<i class="fas fa-search"></i>',
            analyzing: '<i class="fas fa-spinner fa-spin"></i>',
            real: '<i class="fas fa-check-circle"></i>',
            fake: '<i class="fas fa-exclamation-triangle"></i>',
            error: '<i class="fas fa-times-circle"></i>'
        };
        return icons[status] || icons.ready;
    }

    animateConfidenceMeter(confidence) {
        this.elements.confidenceValue.textContent = `${confidence}%`;
        
        // Animate the confidence bar
        setTimeout(() => {
            this.elements.confidenceFill.style.width = `${confidence}%`;
        }, 100);
    }

    showResultsSection() {
        this.elements.resultsSection.style.display = 'block';
        this.elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    showError(message) {
        this.updateResultsUI('error', 'Analysis Failed', 0, message);
        this.showResultsSection();
    }

    setLoadingState(loading) {
        const btn = this.elements.analyzeBtn;
        const btnText = this.elements.btnText;
        const loadingEl = this.elements.btnLoading;

        if (loading) {
            btn.disabled = true;
            btnText.style.opacity = '0';
            loadingEl.style.display = 'block';
        } else {
            btn.disabled = false;
            btnText.style.opacity = '1';
            loadingEl.style.display = 'none';
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    resetAnalysis() {
        this.elements.uploadForm.reset();
        this.elements.filePreview.classList.remove('active');
        this.elements.resultsSection.style.display = 'none';
        this.setInitialState();
    }
}

// Global function for reset button
function resetAnalysis() {
    if (window.app) {
        window.app.resetAnalysis();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DeepfakeDetectorApp();
});