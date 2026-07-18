export class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, options);
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.message || `API error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API Request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    uploadDocument(file, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            const formData = new FormData();
            formData.append('file', file);

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && onProgress) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    onProgress(percent);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch (e) {
                        resolve({ status: 'success' }); // Fallback
                    }
                } else {
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            });

            xhr.addEventListener('error', () => reject(new Error('Network error during upload')));
            xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));

            xhr.open('POST', `${this.baseUrl}/api/docs/upload`);
            xhr.send(formData);
        });
    }

    async listDocuments() {
        return this.request('/api/docs');
    }

    async deleteDocument(docId) {
        return this.request(`/api/docs/${docId}`, { method: 'DELETE' });
    }

    async sendQuery(query, history = [], topK = 5) {
        return this.request('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, history, top_k: topK })
        });
    }

    async getIngestionStatus(docId) {
        return this.request(`/api/status/${docId}`);
    }
}

export const api = new ApiClient();
