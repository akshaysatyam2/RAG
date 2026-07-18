import { api } from './api.js';
import { showToast } from './app.js';

let pollingIntervals = {};

export function initDocuments() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    
    if (uploadZone) {
        uploadZone.addEventListener('click', () => fileInput.click());
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-active');
        });
        
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-active');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-active');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleUpload(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleUpload(e.target.files[0]);
                fileInput.value = ''; // Reset
            }
        });
    }

    refreshDocuments();
}

async function handleUpload(file) {
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const progressText = document.getElementById('upload-progress-text');
    const content = document.querySelector('.upload-content');
    
    content.classList.add('hidden');
    progressContainer.classList.remove('hidden');
    progressBar.style.width = '0%';
    progressText.textContent = '0%';

    try {
        const response = await api.uploadDocument(file, (percent) => {
            progressBar.style.width = `${percent}%`;
            progressText.textContent = `${percent}%`;
        });
        showToast('Document uploaded successfully', 'success');
        refreshDocuments();
        if (response.id) {
            pollStatus(response.id);
        }
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        setTimeout(() => {
            progressContainer.classList.add('hidden');
            content.classList.remove('hidden');
        }, 1000);
    }
}

export async function refreshDocuments() {
    const list = document.getElementById('documents-list');
    if (!list) return;

    try {
        const data = await api.listDocuments();
        const docs = data.documents || [];
        
        if (docs.length === 0) {
            list.innerHTML = '<div style="text-align:center;color:var(--text-secondary);font-size:0.875rem;padding:16px;">No documents yet</div>';
            return;
        }

        list.innerHTML = docs.map(doc => `
            <div class="doc-card" id="doc-${doc.id}">
                <div class="doc-header">
                    <span class="doc-title" title="${doc.name || 'Document'}">${doc.name || 'Unnamed Document'}</span>
                    <button class="delete-btn" data-id="${doc.id}" title="Delete document">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                    </button>
                </div>
                <div class="doc-status">
                    <span class="status-badge status-${doc.status.toLowerCase()}">${doc.status}</span>
                </div>
            </div>
        `).join('');

        list.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                if (confirm('Are you sure you want to delete this document?')) {
                    const id = btn.getAttribute('data-id');
                    try {
                        await api.deleteDocument(id);
                        showToast('Document deleted', 'success');
                        refreshDocuments();
                    } catch (err) {
                        showToast('Failed to delete document', 'error');
                    }
                }
            });
        });

        docs.forEach(doc => {
            if (doc.status.toLowerCase() === 'processing' || doc.status.toLowerCase() === 'pending') {
                pollStatus(doc.id);
            }
        });

    } catch (error) {
        console.error('Failed to list documents:', error);
    }
}

function pollStatus(docId) {
    if (pollingIntervals[docId]) return;
    
    pollingIntervals[docId] = setInterval(async () => {
        try {
            const statusData = await api.getIngestionStatus(docId);
            const badge = document.querySelector(`#doc-${docId} .status-badge`);
            if (badge) {
                badge.className = `status-badge status-${statusData.status.toLowerCase()}`;
                badge.textContent = statusData.status;
            }
            
            if (statusData.status.toLowerCase() !== 'processing' && statusData.status.toLowerCase() !== 'pending') {
                clearInterval(pollingIntervals[docId]);
                delete pollingIntervals[docId];
                if (statusData.status.toLowerCase() === 'ready') {
                    showToast('Document processing complete', 'success');
                }
            }
        } catch (error) {
            clearInterval(pollingIntervals[docId]);
            delete pollingIntervals[docId];
        }
    }, 3000);
}
