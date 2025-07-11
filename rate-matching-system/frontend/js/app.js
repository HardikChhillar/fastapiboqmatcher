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
        console.log('Selected file:', file); // Debug line
        
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
        
        // Clear any previous error messages
        showStatus('query-status', '', 'success');
        
    } catch (error) {
        console.error('Query file selection error:', error);
        App.state.selectedQueryFile = null;
        document.querySelector('#query-tab .upload-text').textContent = 'Click to select Excel file';
        document.getElementById('process-query-btn').disabled = true;
        showStatus('query-status', error.message, 'error');
    }
}

// Dataset upload functionality
async function uploadDataset() {
    if (!App.state.selectedDatasetFile) {
        showStatus('dataset-status', 'Please select a file first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', App.state.selectedDatasetFile);

    const btn = document.getElementById('upload-dataset-btn');
    const btnText = document.getElementById('upload-dataset-text');
    const progressContainer = document.getElementById('dataset-progress');
    const progressFill = document.getElementById('dataset-progress-fill');

    try {
        btn.disabled = true;
        btnText.innerHTML = '<span class="loading"></span>Uploading...';
        progressContainer.classList.remove('hidden');
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }, 200);

        const response = await fetch(`${App.config.API_BASE_URL}/update-dataset/`, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        showStatus('dataset-status', 
            `✅ Dataset updated successfully! ${data.updates_made} entries updated, ${data.new_entries} new entries added. Total dataset size: ${data.total_dataset_size}`, 
            'success'
        );
        
        logActivity(`Dataset updated: ${data.updates_made} updates, ${data.new_entries} new entries`);
        refreshDashboard();

    } catch (error) {
        console.error('Upload error:', error);
        showStatus('dataset-status', '❌ Upload failed: ' + error.message, 'error');
        logActivity(`Dataset upload failed: ${error.message}`);
        
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Upload Dataset';
        setTimeout(() => {
            progressContainer.classList.add('hidden');
            progressFill.style.width = '0%';
        }, 2000);
    }
}

// Query processing functionality
async function processQueryFile() {
    console.log('ProcessQueryFile called, selected file:', App.state.selectedQueryFile); // Debug line
    
    if (!App.state.selectedQueryFile) {
        showStatus('query-status', 'Please select a file first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', App.state.selectedQueryFile);

    const btn = document.getElementById('process-query-btn');
    const btnText = document.getElementById('process-query-text');
    const progressContainer = document.getElementById('query-progress');
    const progressFill = document.getElementById('query-progress-fill');

    try {
        btn.disabled = true;
        btnText.innerHTML = '<span class="loading"></span>Processing...';
        progressContainer.classList.remove('hidden');
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 85) progress = 85;
            progressFill.style.width = progress + '%';
        }, 300);

        const response = await fetch(`${App.config.API_BASE_URL}/query-rate/`, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Get metadata from response headers
        const ratesFilled = response.headers.get('x-rates-filled') || '0';
        const totalRows = response.headers.get('x-total-rows') || '0';

        // Create download link
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = App.state.selectedQueryFile.name.replace(/\.[^/.]+$/, '') + '_filled_rates.xlsx';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        showStatus('query-status', 
            `✅ Query processed successfully! ${ratesFilled} rates filled out of ${totalRows} total rows. File downloaded automatically.`, 
            'success'
        );
        
        logActivity(`Query processed: ${ratesFilled}/${totalRows} rates filled`);
        refreshDashboard();

    } catch (error) {
        console.error('Query error:', error);
        showStatus('query-status', '❌ Query processing failed: ' + error.message, 'error');
        logActivity(`Query processing failed: ${error.message}`);
        
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Process Query';
        setTimeout(() => {
            progressContainer.classList.add('hidden');
            progressFill.style.width = '0%';
        }, 2000);
    }
}

// Dashboard functionality
async function refreshDashboard() {
    const refreshBtn = document.getElementById('refresh-text');
    const originalText = refreshBtn ? refreshBtn.textContent : 'Refresh Data';
    
    try {
        if (refreshBtn) {
            refreshBtn.innerHTML = '<span class="loading"></span>Refreshing...';
        }
        
        // Get dataset info
        const response = await fetch(`${App.config.API_BASE_URL}/dataset-info/`);
        const datasetData = await response.json();
        
        // Get token usage
        const tokenResponse = await fetch(`${App.config.API_BASE_URL}/token-usage/`);
        const tokenData = await tokenResponse.json();
        
        // Safely update dashboard with fallbacks
        const updateElement = (id, value, defaultValue = 0) => {
            const element = document.getElementById(id);
            if (element) {
                const num = parseInt(value) || defaultValue;
                element.textContent = num.toLocaleString();
            }
        };

        updateElement('total-entries', datasetData.total_entries);
        updateElement('tokens-used', tokenData.daily_tokens_used);
        updateElement('cached-matches', tokenData.cached_matches);
        updateElement('total-descriptions', datasetData.total_descriptions);
        
        // Handle average descriptions which might be a decimal
        const avgElement = document.getElementById('avg-descriptions');
        if (avgElement) {
            const avg = parseFloat(datasetData.average_descriptions_per_entry) || 0;
            avgElement.textContent = avg.toFixed(2);
        }
        
        logActivity('Dashboard refreshed successfully');
        
    } catch (error) {
        console.error('Dashboard refresh error:', error);
        logActivity('Dashboard refresh failed: ' + (error.message || 'Unknown error'));
    } finally {
        if (refreshBtn) {
            refreshBtn.textContent = originalText;
        }
    }
}

async function clearCache() {
    try {
        const response = await fetch(`${App.config.API_BASE_URL}/clear-cache/`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        showStatus('dashboard-status', '✅ Cache cleared successfully!', 'success');
        logActivity('Cache cleared successfully');
        refreshDashboard();
    } catch (error) {
        showStatus('dashboard-status', '❌ Failed to clear cache: ' + error.message, 'error');
        logActivity('Cache clear failed: ' + error.message);
    }
}

async function refreshDescriptions() {
    const refreshBtn = document.getElementById('refresh-descriptions-text');
    const originalText = refreshBtn ? refreshBtn.textContent : 'Refresh Descriptions';
    
    try {
        if (refreshBtn) {
            refreshBtn.innerHTML = '<span class="loading"></span>Updating...';
        }
        
        const response = await fetch(`${App.config.API_BASE_URL}/refresh-descriptions/`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        showStatus('dashboard-status', '✅ Successfully refreshed descriptions with construction industry synonyms', 'success');
        logActivity('Descriptions refreshed successfully');
        
        // Refresh dashboard to show updated counts
        await refreshDashboard();
        
    } catch (error) {
        console.error('Description refresh error:', error);
        showStatus('dashboard-status', '❌ Failed to refresh descriptions: ' + error.message, 'error');
        logActivity('Description refresh failed: ' + error.message);
    } finally {
        if (refreshBtn) {
            refreshBtn.textContent = originalText;
        }
    }
}

// Entries Modal Functions
async function showEntriesModal() {
    const modal = document.getElementById('entries-modal');
    if (modal) {
        modal.style.display = 'block';
        
        const tbody = document.getElementById('entries-table-body');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="3" style="text-align: center;"><div class="loading"></div> Loading entries...</td></tr>';
        }
        
        try {
            const response = await fetch(`${App.config.API_BASE_URL}/dataset-entries/`);
            if (response.ok) {
                const data = await response.json();
                if (data && Array.isArray(data.entries)) {
                    App.state.allEntries = data.entries;
                    
                    // Update counts
                    const totalEntriesEl = document.getElementById('total-entries');
                    const totalCountEl = document.getElementById('total-count');
                    if (totalEntriesEl) totalEntriesEl.textContent = App.state.allEntries.length.toLocaleString();
                    if (totalCountEl) totalCountEl.textContent = App.state.allEntries.length.toLocaleString();
                    
                    // Get unique units for filter
                    App.state.uniqueUnits = new Set(App.state.allEntries.map(entry => entry.unit).filter(Boolean));
                    populateUnitFilter();
                    
                    // Initial display
                    filterEntries();
                } else {
                    throw new Error('Invalid response format');
                }
            } else {
                throw new Error('Network response was not ok');
            }
        } catch (error) {
            console.error('Error fetching entries:', error);
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="3" style="text-align: center; padding: 20px;">
                            <div style="color: #dc3545; margin-bottom: 10px;">❌ Failed to load entries</div>
                            <div style="color: #666; font-size: 0.9em;">${error.message || 'Please try again later'}</div>
                        </td>
                    </tr>
                `;
            }
        }
    }
}

function closeEntriesModal() {
    const modal = document.getElementById('entries-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function populateUnitFilter() {
    const unitFilter = document.getElementById('unit-filter');
    if (unitFilter) {
        unitFilter.innerHTML = '<option value="">All Units</option>';
        
        Array.from(App.state.uniqueUnits).sort().forEach(unit => {
            const option = document.createElement('option');
            option.value = unit;
            option.textContent = unit;
            unitFilter.appendChild(option);
        });
    }
}

function filterEntries() {
    const searchInput = document.getElementById('search-input');
    const unitFilter = document.getElementById('unit-filter');
    const rateFilter = document.getElementById('rate-filter');
    
    const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
    const unitFilterValue = unitFilter ? unitFilter.value : '';
    const rateFilterValue = rateFilter ? rateFilter.value : '';
    
    let filtered = App.state.allEntries.filter(entry => {
        const description = entry.description || '';
        const matchesSearch = description.toLowerCase().includes(searchTerm);
        
        const matchesUnit = !unitFilterValue || entry.unit === unitFilterValue;
        const matchesRate = matchesRateFilter(entry.rate, rateFilterValue);
        
        return matchesSearch && matchesUnit && matchesRate;
    });
    
    displayEntries(filtered);
    
    // Update counts
    const filteredCountEl = document.getElementById('filtered-count');
    const totalCountEl = document.getElementById('total-count');
    if (filteredCountEl) filteredCountEl.textContent = filtered.length.toLocaleString();
    if (totalCountEl) totalCountEl.textContent = App.state.allEntries.length.toLocaleString();
}

function matchesRateFilter(rate, filter) {
    if (!filter) return true;
    
    const rateNum = parseFloat(rate);
    switch(filter) {
        case '0-100': return rateNum >= 0 && rateNum <= 100;
        case '100-500': return rateNum > 100 && rateNum <= 500;
        case '500-1000': return rateNum > 500 && rateNum <= 1000;
        case '1000+': return rateNum > 1000;
        default: return true;
    }
}

function displayEntries(entries) {
    const tbody = document.getElementById('entries-table-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (entries.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="3" style="text-align: center; padding: 20px;">
                    <div style="color: #666;">No matching entries found</div>
                </td>
            </tr>
        `;
        return;
    }
    
    entries.forEach(entry => {
        const row = document.createElement('tr');
        
        const displayDescription = entry.description || '';
        
        row.innerHTML = `
            <td>${displayDescription}</td>
            <td>${entry.unit || ''}</td>
            <td>${entry.rate ? '₹' + parseFloat(entry.rate).toLocaleString() : ''}</td>
        `;
        tbody.appendChild(row);
    });
}

// Match Logs Modal Functions
async function showMatchLogsModal() {
    const modal = document.getElementById('match-logs-modal');
    if (modal) {
        modal.style.display = 'block';
        
        try {
            const response = await fetch(`${App.config.API_BASE_URL}/match-logs/`);
            const data = await response.json();
            App.state.allMatchLogs = data;
            
            // Update total count
            const totalMatchesEl = document.getElementById('total-matches-count');
            if (totalMatchesEl) {
                totalMatchesEl.textContent = App.state.allMatchLogs.length.toLocaleString();
            }
            
            // Initial display
            filterMatchLogs();
            
        } catch (error) {
            console.error('Error fetching match logs:', error);
            const tbody = document.getElementById('match-logs-table-body');
            if (tbody) {
                tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: red;">❌ Failed to load match logs: ${error.message}</td></tr>`;
            }
        }
    }
}

function closeMatchLogsModal() {
    const modal = document.getElementById('match-logs-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function filterMatchLogs() {
    // This function would be implemented based on your needs
    // For now, just display all logs
    displayMatchLogs(App.state.allMatchLogs);
}

function displayMatchLogs(logs) {
    const tbody = document.getElementById('match-logs-table-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (logs.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; padding: 20px;">
                    <div style="color: #666;">No match logs found</div>
                </td>
            </tr>
        `;
        return;
    }
    
    logs.forEach(log => {
        const row = document.createElement('tr');
        const timestamp = new Date(log.timestamp).toLocaleString();
        
        row.innerHTML = `
            <td>${timestamp}</td>
            <td class="description-cell">${log.query_description}</td>
            <td class="description-cell">${log.matched_description}</td>
            <td class="unit-cell">
                ${log.query_unit}
                ${log.unit_mismatch ? `<span class="unit-mismatch">(≠ ${log.matched_unit})</span>` : ''}
            </td>
            <td>₹${parseFloat(log.rate).toLocaleString()}</td>
            <td><span class="status-badge status-${log.status}">${log.status}</span></td>
            <td>-</td>
        `;
        tbody.appendChild(row);
    });
}

function sortMatchLogs(field) {
    // Basic sorting implementation
    if (App.state.currentMatchSort.field === field) {
        App.state.currentMatchSort.direction = App.state.currentMatchSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        App.state.currentMatchSort.field = field;
        App.state.currentMatchSort.direction = 'asc';
    }
    
    App.state.allMatchLogs.sort((a, b) => {
        let aVal = a[field];
        let bVal = b[field];
        
        if (field === 'timestamp') {
            aVal = new Date(aVal);
            bVal = new Date(bVal);
        }
        
        if (App.state.currentMatchSort.direction === 'asc') {
            return aVal > bVal ? 1 : -1;
        } else {
            return aVal < bVal ? 1 : -1;
        }
    });
    
    displayMatchLogs(App.state.allMatchLogs);
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