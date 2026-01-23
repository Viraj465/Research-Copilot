const API_BASE_URL = 'https://viraj0112-research-copilot.hf.space/api';

export const api = {
    token: null,

    setAuthToken(token) {
        this.token = token;
    },

    getHeaders(contentType = 'application/json') {
        const headers = {
            'Content-Type': contentType,
        };
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        return headers;
    },

    /**
     * Start a new research session with a URL
     */
    async startSessionUrl(url, llmConfig) {
        // Import and add API keys to llmConfig
        const { getApiKeys } = await import('./config.js');
        const apiKeys = getApiKeys();

        const configWithKeys = {
            ...llmConfig,
            api_keys: apiKeys
        };

        const response = await fetch(`${API_BASE_URL}/sessions/start-url`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                paper_url: url,
                llm_config: configWithKeys
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start session');
        }

        return response.json();
    },

    /**
     * Start a new research session with a PDF upload
     */
    async startSessionUpload(file, llmConfig) {
        const { getApiKeys } = await import('./config.js');
        const apiKeys = getApiKeys();

        const formData = new FormData();
        formData.append('file', file);

        const configWithKeys = {
            ...llmConfig,
            api_keys: apiKeys
        };

        if (configWithKeys) {
            if (configWithKeys.provider) formData.append('llm_provider', configWithKeys.provider);
            if (configWithKeys.model) formData.append('llm_model', configWithKeys.model);
            if (configWithKeys.api_key) formData.append('llm_api_key', configWithKeys.api_key);
            if (configWithKeys.api_keys) formData.append('llm_api_keys', JSON.stringify(configWithKeys.api_keys));
            if (configWithKeys.agents) formData.append('llm_agents', JSON.stringify(configWithKeys.agents));
        }

        const headers = {};
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        const response = await fetch(`${API_BASE_URL}/sessions/start-upload`, {
            method: 'POST',
            headers: headers,
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            console.error('Upload API error:', error);
            throw new Error(error.detail || 'Failed to upload paper');
        }

        const data = await response.json();
        console.log('Upload API response:', data);
        return data;
    },

    /**
     * Get session status and messages
     */
    async getSession(sessionId) {
        const headers = this.token ? { 'Authorization': `Bearer ${this.token}` } : {};
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, { headers });
        if (!response.ok) throw new Error('Failed to get session');
        return response.json();
    },

    /**
     * Get final report
     */
    async getReport(sessionId) {
        const headers = this.token ? { 'Authorization': `Bearer ${this.token}` } : {};
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/report`, { headers });
        if (!response.ok) throw new Error('Failed to get report');
        return response.json();
    },

    /**
     * Send a deep dive chat message
     */
    async sendChatMessage(sessionId, field, message) {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/chat`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                field: field,
                message: message
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to send message');
        }

        return response.json();
    },

    /**
     * Get EventSource for streaming updates
     */
    getEventSource(sessionId) {
        // EventSource doesn't support headers natively. 
        // We might need to pass token in query param or use a polyfill.
        // For now, let's assume the stream endpoint is public or uses a query param if needed.
        // Security Note: Passing token in URL is not ideal but standard EventSource limitation.
        const url = new URL(`${API_BASE_URL}/sessions/${sessionId}/stream`);
        if (this.token) {
            url.searchParams.append('token', this.token);
        }
        console.log('📡 Creating EventSource for URL:', url.toString());
        const es = new EventSource(url.toString());
        console.log('📡 EventSource created with readyState:', es.readyState);
        return es;
    }
};
