// Study Assistant Frontend JavaScript

class StudyAssistant {
    constructor() {
        this.documents = [];
        this.currentAnswer = null;
        this.preferences = this.loadPreferences();
        this.initializeEventListeners();
        this.loadDocuments();
        this.initializePreferences();
        this.hideGuideIfSeen();
    }

    initializeEventListeners() {
        // File upload - enhanced drag and drop
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Question asking - enhanced
        const askButton = document.getElementById('askButton');
        const questionInput = document.getElementById('questionInput');
        const clearButton = document.getElementById('clearButton');
        
        askButton.addEventListener('click', this.askQuestion.bind(this));
        clearButton.addEventListener('click', () => {
            questionInput.value = '';
            questionInput.focus();
        });
        
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.askQuestion();
            }
        });
        
        // Document search
        const documentSearch = document.getElementById('documentSearch');
        if (documentSearch) {
            documentSearch.addEventListener('input', this.filterDocuments.bind(this));
        }
        
        // Preferences
        const responseStyle = document.getElementById('responseStyle');
        const maxChunks = document.getElementById('maxChunks');
        
        if (responseStyle) {
            responseStyle.addEventListener('change', this.savePreferences.bind(this));
        }
        if (maxChunks) {
            maxChunks.addEventListener('change', this.savePreferences.bind(this));
        }
        
        // Tag input handling
        document.addEventListener('keydown', (e) => {
            const newTagInput = document.getElementById('newTagInput');
            if (document.activeElement === newTagInput && e.key === 'Enter') {
                e.preventDefault();
                this.addTag();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
        
        // Auto-hide loading states
        this.setupAutoHideStates();
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        // Only remove dragover if we're leaving the upload area entirely
        if (!e.currentTarget.contains(e.relatedTarget)) {
            document.getElementById('uploadArea').classList.remove('dragover');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadArea').classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        const pdfFiles = files.filter(file => file.name.toLowerCase().endsWith('.pdf'));
        
        if (pdfFiles.length === 0) {
            this.showUploadStatus('Please drop PDF files only', 'error');
            return;
        }
        
        if (pdfFiles.length === 1) {
            this.uploadFile(pdfFiles[0]);
        } else {
            this.uploadMultipleFiles(pdfFiles);
        }
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        
        if (files.length === 1) {
            this.uploadFile(files[0]);
        } else {
            this.uploadMultipleFiles(files);
        }
        
        // Reset file input
        e.target.value = '';
    }

    async uploadMultipleFiles(files) {
        this.showLoadingOverlay('Uploading multiple files...', `Processing ${files.length} files`);
        
        let successCount = 0;
        let errorCount = 0;
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            this.updateLoadingMessage(`Uploading ${file.name} (${i + 1}/${files.length})`);
            
            try {
                await this.uploadFile(file, true); // Silent mode
                successCount++;
            } catch (error) {
                errorCount++;
                console.error(`Failed to upload ${file.name}:`, error);
            }
        }
        
        this.hideLoadingOverlay();
        
        if (successCount > 0) {
            this.showUploadStatus(
                `Successfully uploaded ${successCount} file${successCount > 1 ? 's' : ''}` +
                (errorCount > 0 ? `, ${errorCount} failed` : ''), 
                errorCount > 0 ? 'warning' : 'success'
            );
            this.loadDocuments();
        } else {
            this.showUploadStatus('All uploads failed', 'error');
        }
    }

    async uploadFile(file, silent = false) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            if (!silent) this.showUploadStatus('Only PDF files are allowed', 'error');
            throw new Error('Invalid file type');
        }

        if (file.size > 50 * 1024 * 1024) {
            if (!silent) this.showUploadStatus('File size exceeds 50MB limit', 'error');
            throw new Error('File too large');
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            if (!silent) {
                this.showUploadStatus('Uploading and processing...', 'info');
                this.showUploadProgress(0);
            }
            
            // Simulate progress for better UX
            const progressInterval = setInterval(() => {
                const progressFill = document.getElementById('progressFill');
                if (progressFill) {
                    const currentWidth = parseInt(progressFill.style.width) || 0;
                    if (currentWidth < 90) {
                        progressFill.style.width = (currentWidth + Math.random() * 10) + '%';
                    }
                }
            }, 200);

            const response = await fetch('/upload-pdf', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            
            const result = await response.json();

            if (response.ok) {
                if (!silent) {
                    this.showUploadProgress(100);
                    setTimeout(() => {
                        this.showUploadStatus(`✅ Successfully uploaded ${result.filename}`, 'success');
                        this.hideUploadProgress();
                    }, 500);
                    this.loadDocuments(); // Refresh document list
                }
                return result;
            } else {
                if (!silent) {
                    this.hideUploadProgress();
                    this.showUploadStatus(result.detail || 'Upload failed', 'error');
                }
                throw new Error(result.detail || 'Upload failed');
            }
        } catch (error) {
            if (!silent) {
                this.hideUploadProgress();
                this.showUploadStatus('Upload failed: ' + error.message, 'error');
            }
            throw error;
        }
    }

    showUploadStatus(message, type) {
        const statusDiv = document.getElementById('uploadStatus');
        statusDiv.innerHTML = message;
        statusDiv.className = type;
        statusDiv.style.display = 'block';
        
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 4000);
        }
    }

    showUploadProgress(percentage) {
        const progressDiv = document.getElementById('uploadProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        if (progressDiv && progressFill && progressText) {
            progressDiv.style.display = 'block';
            progressFill.style.width = percentage + '%';
            
            if (percentage < 30) {
                progressText.textContent = 'Uploading file...';
            } else if (percentage < 70) {
                progressText.textContent = 'Processing PDF...';
            } else if (percentage < 90) {
                progressText.textContent = 'Generating embeddings...';
            } else {
                progressText.textContent = 'Finalizing...';
            }
        }
    }

    hideUploadProgress() {
        const progressDiv = document.getElementById('uploadProgress');
        if (progressDiv) {
            setTimeout(() => {
                progressDiv.style.display = 'none';
                const progressFill = document.getElementById('progressFill');
                if (progressFill) progressFill.style.width = '0%';
            }, 1000);
        }
    }

    async loadDocuments() {
        try {
            const response = await fetch('/documents');
            const data = await response.json();
            
            this.documents = data.documents;
            this.renderDocuments(this.documents);
            this.updateDocumentStats();
            
        } catch (error) {
            console.error('Failed to load documents:', error);
            this.showUploadStatus('Failed to load documents', 'error');
        }
    }

    renderDocuments(documents) {
        const documentsList = document.getElementById('documentsList');
        
        if (documents.length === 0) {
            documentsList.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📖</div>
                    <h3>No documents uploaded yet</h3>
                    <p>Upload your first PDF to get started with AI-powered studying</p>
                </div>
            `;
        } else {
            documentsList.innerHTML = documents.map(doc => `
                <div class="document-item" data-filename="${doc.filename.toLowerCase()}" data-tags="${(doc.tags || []).join(' ').toLowerCase()}">
                    <div class="document-info">
                        <h4>${this.escapeHtml(doc.filename)}</h4>
                        <p>📅 Uploaded: ${new Date(doc.upload_date).toLocaleDateString()}</p>
                        <p>📄 ${doc.page_count || 'Unknown'} pages • 🔍 ${doc.chunk_count} searchable chunks</p>
                        <p>💾 ${this.formatFileSize(doc.file_size)}</p>
                        ${doc.tags && doc.tags.length > 0 ? `
                            <div class="document-tags">
                                🏷️ ${doc.tags.map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
                            </div>
                        ` : ''}
                    </div>
                    <div class="document-actions">
                        <button class="tag-btn" onclick="app.showTagModal('${doc.id}', '${this.escapeHtml(doc.filename)}', ${JSON.stringify(doc.tags || []).replace(/"/g, '&quot;')})" title="Manage Tags">
                            🏷️ Tags
                        </button>
                        <button class="delete-btn" onclick="app.deleteDocument('${doc.id}', '${this.escapeHtml(doc.filename)}')">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
            `).join('');
        }
    }

    filterDocuments() {
        const searchTerm = document.getElementById('documentSearch').value.toLowerCase();
        const filteredDocs = this.documents.filter(doc => {
            const filenameMatch = doc.filename.toLowerCase().includes(searchTerm);
            const tagMatch = doc.tags && doc.tags.some(tag => tag.toLowerCase().includes(searchTerm));
            return filenameMatch || tagMatch;
        });
        this.renderDocuments(filteredDocs);
        this.updateDocumentStats(filteredDocs.length);
    }

    updateDocumentStats(filteredCount = null) {
        const statsElement = document.getElementById('documentStats');
        if (statsElement) {
            const count = filteredCount !== null ? filteredCount : this.documents.length;
            const total = this.documents.length;
            
            if (filteredCount !== null && filteredCount !== total) {
                statsElement.textContent = `${count} of ${total} documents`;
            } else {
                statsElement.textContent = `${count} document${count !== 1 ? 's' : ''}`;
            }
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async deleteDocument(docId, filename) {
        if (!confirm(`Are you sure you want to delete "${filename}"?\n\nThis action cannot be undone.`)) {
            return;
        }

        try {
            this.showLoadingOverlay('Deleting document...', 'Removing from database and search index');
            
            const response = await fetch(`/documents/${docId}`, {
                method: 'DELETE'
            });

            const result = await response.json();
            
            this.hideLoadingOverlay();

            if (response.ok) {
                this.showUploadStatus(`✅ Successfully deleted "${filename}"`, 'success');
                this.loadDocuments(); // Refresh document list
            } else {
                this.showUploadStatus(`Failed to delete document: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.hideLoadingOverlay();
            this.showUploadStatus('Failed to delete document: ' + error.message, 'error');
        }
    }

    async askQuestion() {
        const questionInput = document.getElementById('questionInput');
        const askButton = document.getElementById('askButton');
        const answerSection = document.getElementById('answerSection');
        const answerContent = document.getElementById('answerContent');
        const sourcesContent = document.getElementById('sourcesContent');
        const processingTime = document.getElementById('processingTime');

        const question = questionInput.value.trim();
        if (!question) {
            this.showNotification('Please enter a question', 'warning');
            questionInput.focus();
            return;
        }

        if (this.documents.length === 0) {
            this.showNotification('Please upload some PDF documents first', 'warning');
            return;
        }

        try {
            // Update button state
            askButton.disabled = true;
            const btnText = askButton.querySelector('.btn-text');
            const btnLoading = askButton.querySelector('.btn-loading');
            btnText.style.display = 'none';
            btnLoading.style.display = 'flex';
            
            // Get preferences
            const responseStyle = document.getElementById('responseStyle').value;
            const maxChunks = parseInt(document.getElementById('maxChunks').value);
            
            const startTime = Date.now();
            
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    max_chunks: maxChunks,
                    response_style: responseStyle
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.currentAnswer = result;
                
                // Format and display answer with highlighting
                answerContent.innerHTML = this.formatAnswer(result.answer);
                
                // Display sources with better formatting
                if (result.sources && result.sources.length > 0) {
                    sourcesContent.innerHTML = `
                        <strong>📚 Sources:</strong><br>
                        ${result.sources.map((source, index) => 
                            `<span class="source-item">${index + 1}. ${source}</span>`
                        ).join('<br>')}
                    `;
                } else {
                    sourcesContent.innerHTML = '<strong>📚 Sources:</strong> No specific sources found';
                }
                
                // Show processing time
                const totalTime = (Date.now() - startTime) / 1000;
                processingTime.innerHTML = `⏱️ Processed in ${totalTime.toFixed(2)}s`;
                
                // Show answer section with animation
                answerSection.style.display = 'block';
                answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Save to recent questions (localStorage)
                this.saveRecentQuestion(question, result);
                
            } else {
                this.showNotification('Failed to get answer: ' + (result.detail || 'Unknown error'), 'error');
            }
        } catch (error) {
            this.showNotification('Failed to get answer: ' + error.message, 'error');
        } finally {
            // Reset button state
            askButton.disabled = false;
            const btnText = askButton.querySelector('.btn-text');
            const btnLoading = askButton.querySelector('.btn-loading');
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
        }
    }

    formatAnswer(answer) {
        // Enhanced answer formatting with study-focused features
        let formatted = answer;
        
        // Format special study elements first
        formatted = formatted.replace(/\*\*📖 Definition:\*\*/g, '<div class="definition-box"><strong>📖 Definition:</strong>');
        formatted = formatted.replace(/\*\*💡 Example:\*\*/g, '<div class="example-box"><strong>💡 Example:</strong>');
        formatted = formatted.replace(/\*\*🔑 ([^*]+)\*\*/g, '<span class="key-concept">🔑 $1</span>');
        
        // Convert remaining markdown-style formatting
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Format bullet points with proper structure
        const lines = formatted.split('\n');
        let inList = false;
        let processedLines = [];
        
        for (let line of lines) {
            const trimmedLine = line.trim();
            
            if (trimmedLine.startsWith('•')) {
                if (!inList) {
                    processedLines.push('<ul class="study-list">');
                    inList = true;
                }
                processedLines.push(`<li>${trimmedLine.substring(1).trim()}</li>`);
            } else if (trimmedLine.match(/^\d+\./)) {
                if (!inList) {
                    processedLines.push('<ol class="study-list">');
                    inList = true;
                }
                processedLines.push(`<li>${trimmedLine.replace(/^\d+\.\s*/, '')}</li>`);
            } else {
                if (inList) {
                    processedLines.push(inList === 'ul' ? '</ul>' : '</ol>');
                    inList = false;
                }
                if (trimmedLine) {
                    processedLines.push(trimmedLine);
                }
            }
        }
        
        if (inList) {
            processedLines.push('</ul>');
        }
        
        formatted = processedLines.join('\n');
        
        // Close definition and example boxes
        formatted = formatted.replace(/(<div class="definition-box">.*?)(?=<div|$)/gs, '$1</div>');
        formatted = formatted.replace(/(<div class="example-box">.*?)(?=<div|$)/gs, '$1</div>');
        
        // Format horizontal rules
        formatted = formatted.replace(/^---$/gm, '<hr class="study-divider">');
        
        // Add paragraphs for better structure
        formatted = formatted.replace(/\n\n/g, '</p><p>');
        formatted = '<p>' + formatted + '</p>';
        
        // Clean up empty paragraphs and fix nested elements
        formatted = formatted.replace(/<p><\/p>/g, '');
        formatted = formatted.replace(/<p>(<div|<ul|<ol|<hr)/g, '$1');
        formatted = formatted.replace(/(<\/div>|<\/ul>|<\/ol>|<hr[^>]*>)<\/p>/g, '$1');
        
        return formatted;
    }

    // New utility methods for enhanced features
    
    loadPreferences() {
        const saved = localStorage.getItem('studyAssistantPreferences');
        return saved ? JSON.parse(saved) : {
            responseStyle: 'detailed',
            maxChunks: 3,
            highlightKeyTerms: true,
            includeExamples: true,
            showPageReferences: true,
            guideHidden: false
        };
    }

    savePreferences() {
        const responseStyle = document.getElementById('responseStyle').value;
        const maxChunks = document.getElementById('maxChunks').value;
        
        this.preferences = {
            ...this.preferences,
            responseStyle,
            maxChunks: parseInt(maxChunks)
        };
        
        localStorage.setItem('studyAssistantPreferences', JSON.stringify(this.preferences));
    }

    initializePreferences() {
        document.getElementById('responseStyle').value = this.preferences.responseStyle;
        document.getElementById('maxChunks').value = this.preferences.maxChunks;
        
        // Initialize modal preferences if they exist
        const responseStylePref = document.querySelector(`input[name="responseStylePref"][value="${this.preferences.responseStyle}"]`);
        if (responseStylePref) responseStylePref.checked = true;
        
        const maxChunksPref = document.getElementById('maxChunksPref');
        if (maxChunksPref) maxChunksPref.value = this.preferences.maxChunks;
        
        const highlightKeyTerms = document.getElementById('highlightKeyTerms');
        if (highlightKeyTerms) highlightKeyTerms.checked = this.preferences.highlightKeyTerms;
        
        const includeExamples = document.getElementById('includeExamples');
        if (includeExamples) includeExamples.checked = this.preferences.includeExamples;
        
        const showPageReferences = document.getElementById('showPageReferences');
        if (showPageReferences) showPageReferences.checked = this.preferences.showPageReferences;
    }

    hideGuideIfSeen() {
        if (this.preferences.guideHidden) {
            const guideSection = document.getElementById('guideSection');
            if (guideSection) {
                guideSection.style.display = 'none';
            }
        }
    }

    saveRecentQuestion(question, answer) {
        const recent = JSON.parse(localStorage.getItem('recentQuestions') || '[]');
        recent.unshift({
            question,
            answer: answer.answer,
            sources: answer.sources,
            timestamp: new Date().toISOString()
        });
        
        // Keep only last 10 questions
        if (recent.length > 10) {
            recent.splice(10);
        }
        
        localStorage.setItem('recentQuestions', JSON.stringify(recent));
    }

    showLoadingOverlay(title, message) {
        const overlay = document.getElementById('loadingOverlay');
        const titleEl = overlay.querySelector('h3');
        const messageEl = overlay.querySelector('#loadingMessage');
        
        if (titleEl) titleEl.textContent = title;
        if (messageEl) messageEl.textContent = message;
        
        overlay.style.display = 'flex';
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = 'none';
    }

    updateLoadingMessage(message) {
        const messageEl = document.getElementById('loadingMessage');
        if (messageEl) messageEl.textContent = message;
    }

    showNotification(message, type = 'info') {
        // Create a temporary notification
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">&times;</button>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#f8d7da' : type === 'warning' ? '#fff3cd' : '#d1ecf1'};
            color: ${type === 'error' ? '#721c24' : type === 'warning' ? '#856404' : '#0c5460'};
            border: 1px solid ${type === 'error' ? '#f5c6cb' : type === 'warning' ? '#ffeaa7' : '#bee5eb'};
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1001;
            max-width: 400px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + K: Focus question input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            document.getElementById('questionInput').focus();
        }
        
        // Escape: Clear question input
        if (e.key === 'Escape') {
            const questionInput = document.getElementById('questionInput');
            if (document.activeElement === questionInput) {
                questionInput.value = '';
            }
        }
    }

    setupAutoHideStates() {
        // Auto-hide upload status after success
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.target.id === 'uploadStatus' && 
                    mutation.target.classList.contains('success')) {
                    setTimeout(() => {
                        mutation.target.style.display = 'none';
                    }, 4000);
                }
            });
        });
        
        const uploadStatus = document.getElementById('uploadStatus');
        if (uploadStatus) {
            observer.observe(uploadStatus, { attributes: true, attributeFilter: ['class'] });
        }
    }

    // Document tagging methods
    showTagModal(docId, filename, tags) {
        this.currentDocumentId = docId;
        this.currentDocumentTags = Array.isArray(tags) ? [...tags] : [];
        
        document.getElementById('tagDocumentName').textContent = filename;
        this.renderCurrentTags();
        document.getElementById('tagModal').style.display = 'flex';
        document.getElementById('newTagInput').focus();
    }

    renderCurrentTags() {
        const currentTagsDiv = document.getElementById('currentTags');
        if (this.currentDocumentTags.length === 0) {
            currentTagsDiv.innerHTML = '<p class="no-tags">No tags assigned</p>';
        } else {
            currentTagsDiv.innerHTML = this.currentDocumentTags.map(tag => `
                <span class="current-tag">
                    ${this.escapeHtml(tag)}
                    <button onclick="app.removeTag('${this.escapeHtml(tag)}')" class="remove-tag-btn">&times;</button>
                </span>
            `).join('');
        }
    }

    addTag() {
        const input = document.getElementById('newTagInput');
        const tag = input.value.trim();
        
        if (tag && !this.currentDocumentTags.includes(tag)) {
            this.currentDocumentTags.push(tag);
            this.renderCurrentTags();
            input.value = '';
        }
    }

    removeTag(tag) {
        this.currentDocumentTags = this.currentDocumentTags.filter(t => t !== tag);
        this.renderCurrentTags();
    }

    async saveDocumentTags() {
        try {
            const response = await fetch(`/documents/${this.currentDocumentId}/tags`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.currentDocumentTags)
            });

            const result = await response.json();

            if (response.ok) {
                this.showNotification('Tags updated successfully!', 'success');
                this.loadDocuments(); // Refresh document list
                this.closeModal('tagModal');
            } else {
                this.showNotification('Failed to update tags: ' + (result.detail || 'Unknown error'), 'error');
            }
        } catch (error) {
            this.showNotification('Failed to update tags: ' + error.message, 'error');
        }
    }

    // Study preferences methods
    showPreferences() {
        this.initializePreferences();
        document.getElementById('preferencesModal').style.display = 'flex';
    }

    saveStudyPreferences() {
        const responseStyle = document.querySelector('input[name="responseStylePref"]:checked').value;
        const maxChunks = parseInt(document.getElementById('maxChunksPref').value);
        const highlightKeyTerms = document.getElementById('highlightKeyTerms').checked;
        const includeExamples = document.getElementById('includeExamples').checked;
        const showPageReferences = document.getElementById('showPageReferences').checked;

        this.preferences = {
            ...this.preferences,
            responseStyle,
            maxChunks,
            highlightKeyTerms,
            includeExamples,
            showPageReferences
        };

        localStorage.setItem('studyAssistantPreferences', JSON.stringify(this.preferences));
        
        // Update the main form controls
        document.getElementById('responseStyle').value = responseStyle;
        document.getElementById('maxChunks').value = maxChunks;

        this.showNotification('Preferences saved successfully!', 'success');
        this.closeModal('preferencesModal');
    }

    resetPreferences() {
        this.preferences = {
            responseStyle: 'detailed',
            maxChunks: 3,
            highlightKeyTerms: true,
            includeExamples: true,
            showPageReferences: true,
            guideHidden: false
        };

        localStorage.setItem('studyAssistantPreferences', JSON.stringify(this.preferences));
        this.initializePreferences();
        this.showNotification('Preferences reset to defaults', 'success');
    }

    closeModal(modalId) {
        document.getElementById(modalId).style.display = 'none';
    }
}

// Global functions for HTML onclick handlers
function toggleGuide() {
    const guideSection = document.getElementById('guideSection');
    const isHidden = guideSection.style.display === 'none';
    
    guideSection.style.display = isHidden ? 'block' : 'none';
    
    // Save preference
    app.preferences.guideHidden = !isHidden;
    localStorage.setItem('studyAssistantPreferences', JSON.stringify(app.preferences));
}

function showPreferences() {
    app.showPreferences();
}

function addTag() {
    app.addTag();
}

function addSuggestedTag(tag) {
    document.getElementById('newTagInput').value = tag;
    app.addTag();
}

function saveDocumentTags() {
    app.saveDocumentTags();
}

function saveStudyPreferences() {
    app.saveStudyPreferences();
}

function resetPreferences() {
    app.resetPreferences();
}

function setExampleQuestion(question) {
    document.getElementById('questionInput').value = question;
    document.getElementById('questionInput').focus();
}

function copyAnswer() {
    if (app.currentAnswer) {
        navigator.clipboard.writeText(app.currentAnswer.answer).then(() => {
            app.showNotification('Answer copied to clipboard!', 'success');
        }).catch(() => {
            app.showNotification('Failed to copy answer', 'error');
        });
    }
}

function saveAnswer() {
    if (app.currentAnswer) {
        const blob = new Blob([
            `Question: ${document.getElementById('questionInput').value}\n\n`,
            `Answer: ${app.currentAnswer.answer}\n\n`,
            `Sources: ${app.currentAnswer.sources.join(', ')}\n\n`,
            `Generated: ${new Date().toLocaleString()}`
        ], { type: 'text/plain' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'study-answer.txt';
        a.click();
        URL.revokeObjectURL(url);
        
        app.showNotification('Answer saved as file!', 'success');
    }
}

function showHelp() {
    document.getElementById('helpModal').style.display = 'flex';
}

function showAbout() {
    app.showNotification('Study Assistant v1.0 - AI-powered study companion', 'info');
}

function closeModal(modalId) {
    app.closeModal(modalId);
}

// Initialize the app when the page loads
const app = new StudyAssistant();