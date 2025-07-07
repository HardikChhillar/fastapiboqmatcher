// Global configuration
const App = {
    config: {
        API_BASE_URL: 'http://localhost:8000',
        STATIC_URL: 'http://localhost:8000/static'
    },
    state: {
        selectedDatasetFile: null,
        selectedQueryFile: null,
        currentMatchData: null,
        allEntries: [],
        allMatchLogs: [],
        uniqueUnits: new Set(),
        currentMatchSort: { field: 'timestamp', direction: 'desc' },
        currentEntriesSort: { field: 'description', direction: 'asc' }
    }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupFileUploadHandlers();
    refreshDashboard();
    setInterval(refreshDashboard, 30000);
    logActivity('Application started successfully');
});

// File upload handlers
function setupFileUploadHandlers() {
    // Dataset file upload
    const datasetUpload = document.querySelector('#dataset-tab .file-upload');
    if (datasetUpload) {
        datasetUpload.addEventListener('dragover', handleDragOver);
        datasetUpload.addEventListener('dragleave', handleDragLeave);
        datasetUpload.addEventListener('drop', (e) => handleDrop(e, 'dataset'));
    }

    // Query file upload
    const queryUpload = document.querySelector('#query-tab .file-upload');
    if (queryUpload) {
        queryUpload.addEventListener('dragover', handleDragOver);
        queryUpload.addEventListener('dragleave', handleDragLeave);
        queryUpload.addEventListener('drop', (e) => handleDrop(e, 'query'));
    }
}

function isValidExcelFile(file) {
    if (!file || !file.name) return false;
    const fileName = file.name.toLowerCase();
    return fileName.endsWith('.xlsx') || fileName.endsWith('.xls');
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e, type) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    try {
        const files = e.dataTransfer.files;
        if (!files || files.length === 0) {
            throw new Error('No file dropped');
        }

        const file = files[0];
        if (!isValidExcelFile(file)) {
            throw new Error('Invalid file type. Please select an Excel file (.xlsx or .xls)');
        }

        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            throw new Error('File size too large. Maximum size is 10MB');
        }

        if (type === 'dataset') {
            handleDatasetFile({files: [file]});
        } else {
            handleQueryFileSelect({files: [file]});
        }
    } catch (error) {
        console.error('File drop error:', error);
        const uploadText = document.querySelector(`#${type}-tab .upload-text`);
        if (uploadText) {
            uploadText.textContent = 'Click to select Excel file';
        }
        showStatus(type + '-status', error.message, 'error');
    }
}

function handleDatasetFile(input) {
    try {
        const file = input.files ? input.files[0] : null;
        if (!file) {
            throw new Error('No file selected');
        }

        if (!isValidExcelFile(file)) {
            throw new Error('Invalid file type. Please select an Excel file (.xlsx or .xls)');
        }

        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            throw new Error('File size too large. Maximum size is 10MB');
        }

        App.state.selectedDatasetFile = file;
        document.querySelector('#dataset-tab .upload-text').textContent = `Selected: ${file.name}`;
        document.getElementById('upload-dataset-btn').disabled = false;
        logActivity(`Dataset file selected: ${file.name}`);
    } catch (error) {
        console.error('Dataset file selection error:', error);
        App.state.selectedDatasetFile = null;
        document.querySelector('#dataset-tab .upload-text').textContent = 'Click to select Excel file';
        document.getElementById('upload-dataset-btn').disabled = true;
        showStatus('dataset-status', error.message, 'error');
    }
}

function handleQueryFileSelect(input) {
    try {
        const file = input.files ? input.files[0] : null;
        if (!file) {
            throw new Error('No file selected');
        }

        if (!isValidExcelFile(file)) {
            throw new Error('Invalid file type. Please select an Excel file (.xlsx or .xls)');
        }

        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            throw new Error('File size too large. Maximum size is 10MB');
        }

        App.state.selectedQueryFile = file;
        document.querySelector('#query-tab .upload-text').textContent = `Selected: ${file.name}`;
        document.getElementById('process-query-btn').disabled = false;
        logActivity(`Query file selected: ${file.name}`);
    } catch (error) {
        console.error('Query file selection error:', error);
        App.state.selectedQueryFile = null;
        document.querySelector('#query-tab .upload-text').textContent = 'Click to select Excel file';
        document.getElementById('process-query-btn').disabled = true;
        showStatus('query-status', error.message, 'error');
    }
}

// Tab switching functionality
function switchTab(tabName) {
    // Remove active class from all tabs and contents
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab and content
    const selectedTab = document.querySelector(`.tab[onclick*="${tabName}"]`);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    const selectedContent = document.getElementById(tabName + '-tab');
    if (selectedContent) {
        selectedContent.classList.add('active');
    }
    
    logActivity(`Switched to ${tabName} tab`);
}

// Utility functions
function showStatus(elementId, message, type) {
    const statusElement = document.getElementById(elementId);
    if (statusElement) {
        statusElement.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusElement.innerHTML = '';
            }, 5000);
        }
    }
}

function logActivity(message) {
    const logContainer = document.getElementById('activity-log');
    if (logContainer) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<strong>${timestamp}</strong> - ${message}`;
        
        logContainer.insertBefore(logEntry, logContainer.firstChild);
        
        // Keep only last 50 entries
        const entries = logContainer.querySelectorAll('.log-entry');
        if (entries.length > 50) {
            entries[entries.length - 1].remove();
        }
    }
}

// Modal functions
function closeMatchLogsModal() {
    const modal = document.getElementById('match-logs-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Export functions that need to be globally accessible
window.App = App;
window.switchTab = switchTab;
window.handleDatasetFile = handleDatasetFile;
window.handleQueryFileSelect = handleQueryFileSelect;
window.uploadDataset = uploadDataset;
window.processQueryFile = processQueryFile;
window.refreshDashboard = refreshDashboard;
window.clearCache = clearCache;
window.refreshDescriptions = refreshDescriptions;
window.showEntriesModal = showEntriesModal;
window.closeEntriesModal = closeEntriesModal;
window.showMatchLogsModal = showMatchLogsModal;
window.closeMatchLogsModal = closeMatchLogsModal;
window.filterEntries = filterEntries;
window.filterMatchLogs = filterMatchLogs;
window.sortMatchLogs = sortMatchLogs; 