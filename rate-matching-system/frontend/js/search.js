// Search and filtering functionality
const Search = {
    init() {
        // Initialize search event listeners
        document.getElementById('search-input')?.addEventListener('keyup', () => this.filterEntries());
        document.getElementById('unit-filter')?.addEventListener('change', () => this.filterEntries());
        document.getElementById('rate-filter')?.addEventListener('change', () => this.filterEntries());
        
        // Match logs search
        document.getElementById('match-search-input')?.addEventListener('keyup', () => this.filterMatchLogs());
        document.getElementById('status-filter')?.addEventListener('change', () => this.filterMatchLogs());
    },

    async showEntriesModal() {
        const modal = document.getElementById('entries-modal');
        modal.style.display = 'block';
        
        try {
            // Show loading state
            const tbody = document.getElementById('entries-table-body');
            tbody.innerHTML = '<tr><td colspan="3" style="text-align: center;">Loading entries...</td></tr>';
            
            // Fetch all entries using global App configuration
            const response = await fetch(`${App.config.API_BASE_URL}/dataset-entries/`);
            const data = await response.json();
            App.state.allEntries = data.entries;
            
            // Update total entries count
            document.getElementById('total-entries').textContent = App.state.allEntries.length.toLocaleString();
            document.getElementById('total-count').textContent = App.state.allEntries.length.toLocaleString();
            
            // Get unique units for filter
            App.state.uniqueUnits = new Set(App.state.allEntries.map(entry => entry.unit));
            this.populateUnitFilter();
            
            // Initial display
            this.filterEntries();
            
        } catch (error) {
            console.error('Error fetching entries:', error);
            const tbody = document.getElementById('entries-table-body');
            tbody.innerHTML = `<tr><td colspan="3" style="text-align: center; color: red;">❌ Failed to load entries: ${error.message}</td></tr>`;
        }
    },

    closeEntriesModal() {
        document.getElementById('entries-modal').style.display = 'none';
    },

    populateUnitFilter() {
        const unitFilter = document.getElementById('unit-filter');
        unitFilter.innerHTML = '<option value="">All Units</option>';
        
        Array.from(App.state.uniqueUnits).sort().forEach(unit => {
            const option = document.createElement('option');
            option.value = unit;
            option.textContent = unit;
            unitFilter.appendChild(option);
        });
    },

    filterEntries() {
        const searchTerm = document.getElementById('search-input').value.toLowerCase();
        const unitFilter = document.getElementById('unit-filter').value;
        const rateFilter = document.getElementById('rate-filter').value;
        
        let filtered = App.state.allEntries.filter(entry => {
            const description = entry.description || '';
            const matchesSearch = description.toLowerCase().includes(searchTerm);
            
            const matchesUnit = !unitFilter || entry.unit === unitFilter;
            const matchesRate = this.matchesRateFilter(entry.rate, rateFilter);
            
            return matchesSearch && matchesUnit && matchesRate;
        });
        
        this.displayEntries(filtered);
        
        // Update counts
        document.getElementById('filtered-count').textContent = filtered.length.toLocaleString();
        document.getElementById('total-count').textContent = App.state.allEntries.length.toLocaleString();
    },

    matchesRateFilter(rate, filter) {
        if (!filter) return true;
        
        const rateNum = parseFloat(rate);
        switch(filter) {
            case '0-100': return rateNum >= 0 && rateNum <= 100;
            case '100-500': return rateNum > 100 && rateNum <= 500;
            case '500-1000': return rateNum > 500 && rateNum <= 1000;
            case '1000+': return rateNum > 1000;
            default: return true;
        }
    },

    displayEntries(entries) {
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
            
            const description = entry.description || '';
            const searchTerm = document.getElementById('search-input').value.toLowerCase();
            const displayDescription = description.toLowerCase().includes(searchTerm) ? description : '';
            
            row.innerHTML = `
                <td>${displayDescription}</td>
                <td>${entry.unit || ''}</td>
                <td>${entry.rate ? '₹' + parseFloat(entry.rate).toLocaleString() : ''}</td>
            `;
            tbody.appendChild(row);
        });
    },

    async showMatchLogsModal() {
        const modal = document.getElementById('match-logs-modal');
        modal.style.display = 'block';
        
        try {
            const response = await fetch(`${App.config.API_BASE_URL}/match-logs/`);
            const data = await response.json();
            App.state.allMatchLogs = data;
            
            // Update total count
            document.getElementById('total-matches-count').textContent = App.state.allMatchLogs.length.toLocaleString();
            
            // Initial display
            this.filterMatchLogs();
            
        } catch (error) {
            console.error('Error fetching match logs:', error);
            const tbody = document.getElementById('match-logs-table-body');
            tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: red;">❌ Failed to load match logs: ${error.message}</td></tr>`;
        }
    },

    filterMatchLogs() {
        const searchTerm = document.getElementById('match-search-input').value.toLowerCase();
        const statusFilter = document.getElementById('status-filter').value;
        const unitMismatchFilter = document.getElementById('unit-mismatch-filter').value;
        const rateFilter = document.getElementById('match-rate-filter').value;
        
        let filtered = App.state.allMatchLogs.filter(log => {
            const matchesSearch = (
                log.query_description.toLowerCase().includes(searchTerm) ||
                log.matched_description.toLowerCase().includes(searchTerm)
            );
            
            const matchesStatus = !statusFilter || log.status === statusFilter;
            const matchesUnitMismatch = unitMismatchFilter === '' || 
                log.unit_mismatch.toString() === unitMismatchFilter;
            const matchesRate = this.matchesRateFilter(log.rate, rateFilter);
            
            return matchesSearch && matchesStatus && matchesUnitMismatch && matchesRate;
        });
        
        this.displayMatchLogs(filtered);
        
        // Update counts
        document.getElementById('filtered-matches-count').textContent = filtered.length.toLocaleString();
        document.getElementById('total-matches-count').textContent = App.state.allMatchLogs.length.toLocaleString();
    },

    displayMatchLogs(logs) {
        const tbody = document.getElementById('match-logs-table-body');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        if (logs.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; padding: 20px;">
                        <div style="color: #666;">No matching logs found</div>
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
                <td>
                    ${log.unit_mismatch ? 
                        `<button class="btn btn-secondary" onclick="showUnitSelectionModal(${JSON.stringify(log)})">
                            Select Unit
                        </button>` : 
                        '-'
                    }
                </td>
            `;
            tbody.appendChild(row);
        });
    },

    closeMatchLogsModal() {
        document.getElementById('match-logs-modal').style.display = 'none';
    }
};

// Initialize search functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', () => Search.init());

// Export Search object globally
window.Search = Search; 