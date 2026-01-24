import { api } from './api.js';
import { ui } from './ui.js';
import { auth } from './auth.js';
import { saveApiKeysToStorage } from './config.js';

// Debug Logging System
const debugLogs = [];
window.getDebugLogs = () => {
    return debugLogs;
};

function addDebugLog(level, message, data = null) {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = {
        timestamp,
        level,
        message,
        data,
        sessionId: state.sessionId
    };
    debugLogs.push(logEntry);
}

// State
const state = {
    sessionId: null,
    status: 'idle', // idle, processing, completed, error
    llmConfig: null,
    activeTab: 'executive_summary',
    reportData: null,
    currentDeepDiveField: null,
    chatHistory: [],
    eventSource: null
};

// DOM Elements
const views = {
    start: document.getElementById('view-start'),
    processing: document.getElementById('view-processing'),
    report: document.getElementById('view-report')
};

const elements = {
    paperUrl: document.getElementById('paper-url'),
    startUrlBtn: document.getElementById('start-url-btn'),
    paperUpload: document.getElementById('paper-upload'),
    logsContainer: document.getElementById('logs-container'),
    timer: document.getElementById('timer'),
    reportContent: document.getElementById('report-content'),
    reportTabs: document.getElementById('report-tabs'),
    reportTitle: document.getElementById('report-title'),
    configBtn: document.getElementById('config-btn'),
    configModal: document.getElementById('config-modal'),
    closeConfig: document.getElementById('close-config'),
    saveConfig: document.getElementById('save-config'),
    llmProvider: document.getElementById('llm-provider'),
    llmModel: document.getElementById('llm-model'),
    apiKeyGroq: document.getElementById('api-key-groq'),
    apiKeyGoogle: document.getElementById('api-key-google'),
    apiKeyOpenAI: document.getElementById('api-key-openai'),
    apiKeyTavily: document.getElementById('api-key-tavily'),
    deepDiveTrigger: document.getElementById('deep-dive-trigger'),
    chatDrawer: document.getElementById('chat-drawer'),
    closeChat: document.getElementById('close-chat'),
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    chatForm: document.getElementById('chat-form'),
    chatContextName: document.getElementById('chat-context-name'),
    newSessionBtn: document.getElementById('new-session-btn'),
    toggleAdvanced: document.getElementById('toggle-advanced'),
    advancedSettings: document.getElementById('advanced-settings'),
    // Canvas elements
    canvasContent: document.getElementById('canvas-content'),
    canvasCursor: document.getElementById('canvas-cursor'),
    currentAgent: document.getElementById('current-agent'),
    logsToggle: document.getElementById('logs-toggle'),
    logsPanel: document.getElementById('logs-panel'),
    canvasSearch: document.getElementById('canvas-search'),
    liveCanvas: document.getElementById('live-canvas'),
    // Auth
    authContainer: document.getElementById('auth-container'),
    signInBtn: document.getElementById('sign-in-btn'),
    signOutBtn: document.getElementById('sign-out-btn'),
    userEmail: document.getElementById('user-email'),
    userAvatar: document.getElementById('user-avatar'),
    authModal: document.getElementById('auth-modal'),
    closeAuth: document.getElementById('close-auth'),
    authGoogleBtn: document.getElementById('auth-google-btn'),
    authGithubBtn: document.getElementById('auth-github-btn'),
    // Theme & Mobile
    themeToggle: document.getElementById('theme-toggle'),
    toggleSidebar: document.getElementById('toggle-sidebar'),
    reportSidebar: document.getElementById('report-sidebar')
};

// Constants
const MODELS = {
    'groq': ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', "meta-llama/llama-guard-4-12b", 'openai/gpt-oss-120b', 'openai/gpt-oss-20b'],
    'google': ['gemini-2.5-flash', 'gemini-2.5-flash-lite'],
    'openai': ['gpt-4.1-mini', 'gpt-4.1', 'gpt-5.1', 'gpt-5.2']
};

const AGENTS = [
    { id: 'paper_analysis', name: 'Paper Analysis' },
    { id: 'web_research', name: 'Web Research' },
    { id: 'sota_tracker', name: 'SOTA Tracker' },
    { id: 'comparative_analysis', name: 'Comparative Analysis' },
    { id: 'direction_advisor', name: 'Direction Advisor' },
    { id: 'report_generation', name: 'Report Generation' }
];

// Initialization
function init() {
    setupEventListeners();
    lucide.createIcons();
    updateModelOptions();
    renderAgentConfigs();
    initTheme();

    // Initialize Auth
    auth.init();

    // Load saved config (except API keys)
    const savedConfig = localStorage.getItem('llm_config');
    if (savedConfig) {
        const config = JSON.parse(savedConfig);
        elements.llmProvider.value = config.provider;
        updateModelOptions();
        elements.llmModel.value = config.model;

        // Load agent overrides
        if (config.agents) {
            Object.entries(config.agents).forEach(([agentId, agentConfig]) => {
                const providerSelect = document.getElementById(`provider-${agentId}`);
                const modelSelect = document.getElementById(`model-${agentId}`);
                if (providerSelect && modelSelect) {
                    providerSelect.value = agentConfig.provider;
                    updateAgentModelOptions(agentId);
                    modelSelect.value = agentConfig.model;
                }
            });
        }

        state.llmConfig = config;
    } else {
        // Initialize with defaults from UI
        state.llmConfig = {
            provider: elements.llmProvider.value,
            model: elements.llmModel.value,
            agents: {}
        };
    }
}

function setupEventListeners() {
    // Auth
    window.addEventListener('auth:change', handleAuthChange);
    if (elements.signInBtn) elements.signInBtn.addEventListener('click', () => elements.authModal.classList.remove('hidden'));
    if (elements.closeAuth) elements.closeAuth.addEventListener('click', () => elements.authModal.classList.add('hidden'));
    if (elements.signOutBtn) elements.signOutBtn.addEventListener('click', () => auth.signOut());
    if (elements.authGoogleBtn) elements.authGoogleBtn.addEventListener('click', () => auth.signInWithGoogle());
    if (elements.authGithubBtn) elements.authGithubBtn.addEventListener('click', () => auth.signInWithGithub());

    // Config
    elements.configBtn.addEventListener('click', () => elements.configModal.classList.remove('hidden'));
    elements.closeConfig.addEventListener('click', () => elements.configModal.classList.add('hidden'));
    elements.llmProvider.addEventListener('change', updateModelOptions);
    elements.saveConfig.addEventListener('click', saveConfiguration);
    elements.toggleAdvanced.addEventListener('click', () => {
        elements.advancedSettings.classList.toggle('hidden');
    });

    // Start Session
    elements.startUrlBtn.addEventListener('click', handleStartUrl);
    if (elements.paperUpload) {
        elements.paperUpload.addEventListener('change', handleUpload);
    }
    elements.newSessionBtn.addEventListener('click', resetSession);

    // Chat
    elements.deepDiveTrigger.addEventListener('click', openChat);
    elements.closeChat.addEventListener('click', closeChat);
    elements.chatForm.addEventListener('submit', handleChatMessage);

    // Export Report
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', handleExportReport);
    }

    // Canvas: Logs Toggle
    if (elements.logsToggle) {
        elements.logsToggle.addEventListener('click', () => {
            elements.logsPanel.classList.toggle('hidden');
        });
    }

    // Canvas: Search
    if (elements.canvasSearch) {
        elements.canvasSearch.addEventListener('input', handleCanvasSearch);
    }

    // Theme Toggle
    if (elements.themeToggle) {
        elements.themeToggle.addEventListener('click', toggleTheme);
    }

    // Mobile Sidebar Toggle
    if (elements.toggleSidebar) {
        elements.toggleSidebar.addEventListener('click', () => {
            elements.reportSidebar.classList.toggle('hidden');
        });
    }
}

function initTheme() {
    // Check localStorage or system preference
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
}

function toggleTheme() {
    if (document.documentElement.classList.contains('dark')) {
        document.documentElement.classList.remove('dark');
        localStorage.theme = 'light';
    } else {
        document.documentElement.classList.add('dark');
        localStorage.theme = 'dark';
    }
}

function handleAuthChange(e) {
    const user = e.detail.user;
    if (user) {
        // Signed In
        elements.authContainer.classList.remove('hidden');
        elements.signInBtn.classList.add('hidden');
        elements.userEmail.textContent = user.email;
        elements.userAvatar.src = user.user_metadata.avatar_url || `https://ui-avatars.com/api/?name=${user.email}`;

        // Update API token
        api.setAuthToken(auth.getToken());
        elements.authModal.classList.add('hidden');
    } else {
        // Signed Out
        elements.authContainer.classList.add('hidden');
        elements.signInBtn.classList.remove('hidden');
        elements.userEmail.textContent = '';
        elements.userAvatar.src = '';

        // Clear API token
        api.setAuthToken(null);
    }
}

function handleCanvasSearch(e) {
    const query = e.target.value.toLowerCase().trim();
    const content = elements.canvasContent;

    // Remove existing highlights
    content.querySelectorAll('.search-highlight').forEach(el => {
        const text = document.createTextNode(el.textContent);
        el.parentNode.replaceChild(text, el);
    });
    content.normalize();

    if (!query) return;

    // Find and highlight matches
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, null, false);
    const matches = [];

    while (walker.nextNode()) {
        const node = walker.currentNode;
        const idx = node.textContent.toLowerCase().indexOf(query);
        if (idx !== -1) {
            matches.push({ node, idx });
        }
    }

    // Highlight first match and scroll to it
    matches.forEach((match, i) => {
        const range = document.createRange();
        range.setStart(match.node, match.idx);
        range.setEnd(match.node, match.idx + query.length);

        const highlight = document.createElement('mark');
        highlight.className = 'search-highlight';
        range.surroundContents(highlight);

        if (i === 0) {
            highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
}

// Configuration Logic
function updateModelOptions() {
    const provider = elements.llmProvider.value;
    const models = MODELS[provider] || [];
    elements.llmModel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
}

function renderAgentConfigs() {
    elements.advancedSettings.innerHTML = AGENTS.map(agent => `
        <div class="bg-gray-50 dark:bg-slate-900 p-3 rounded border border-gray-200 dark:border-slate-700">
            <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-medium text-gray-700 dark:text-gray-300">${agent.name}</span>
                <span class="text-xs text-gray-400 font-mono">${agent.id}</span>
            </div>
            <div class="grid grid-cols-2 gap-2">
                <select id="provider-${agent.id}" onchange="window.updateAgentModelOptions('${agent.id}')" 
                    class="text-sm px-2 py-1 border border-gray-300 dark:border-slate-600 rounded focus:ring-1 focus:ring-accent outline-none dark:bg-slate-800 dark:text-white">
                    <option value="">Use Global</option>
                    <option value="groq">Groq</option>
                    <option value="google">Google</option>
                    <option value="openai">OpenAI</option>
                </select>
                <select id="model-${agent.id}" disabled
                    class="text-sm px-2 py-1 border border-gray-300 dark:border-slate-600 rounded focus:ring-1 focus:ring-accent outline-none bg-gray-100 dark:bg-slate-800/50 dark:text-gray-400">
                    <option value="">Default Model</option>
                </select>
            </div>
        </div>
    `).join('');
}

window.updateAgentModelOptions = (agentId) => {
    const providerSelect = document.getElementById(`provider-${agentId}`);
    const modelSelect = document.getElementById(`model-${agentId}`);
    const provider = providerSelect.value;

    if (!provider) {
        modelSelect.innerHTML = '<option value="">Default Model</option>';
        modelSelect.disabled = true;
        modelSelect.classList.add('bg-gray-100', 'dark:bg-slate-800/50', 'dark:text-gray-400');
        modelSelect.classList.remove('dark:bg-slate-800', 'dark:text-white');
        return;
    }

    const models = MODELS[provider] || [];
    modelSelect.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
    modelSelect.disabled = false;
    modelSelect.classList.remove('bg-gray-100', 'dark:bg-slate-800/50', 'dark:text-gray-400');
    modelSelect.classList.add('dark:bg-slate-800', 'dark:text-white');
};

function saveConfiguration() {
    const config = {
        provider: elements.llmProvider.value,
        model: elements.llmModel.value,
        api_keys: {
            'groq': elements.apiKeyGroq.value,
            'google': elements.apiKeyGoogle.value,
            'openai': elements.apiKeyOpenAI.value,
            'tavily': elements.apiKeyTavily.value
        },
        agents: {}
    };

    // Collect agent overrides
    AGENTS.forEach(agent => {
        const provider = document.getElementById(`provider-${agent.id}`).value;
        const model = document.getElementById(`model-${agent.id}`).value;

        if (provider && model) {
            config.agents[agent.id] = {
                provider: provider,
                model: model
            };
        }
    });

    state.llmConfig = config;

    // Save API keys to storage (updates config.js and localStorage)
    saveApiKeysToStorage(
        elements.apiKeyTavily.value,
        elements.apiKeyGroq.value,
        elements.apiKeyGoogle.value,
        elements.apiKeyOpenAI.value
    );

    // Save to local storage (excluding API keys for security)
    const safeConfig = { ...config, api_keys: {} };
    localStorage.setItem('llm_config', JSON.stringify(safeConfig));

    elements.configModal.classList.add('hidden');
    alert('Configuration saved!');
}

async function handleStartUrl() {
    const url = elements.paperUrl.value.trim();
    if (!url) return alert('Please enter a URL');

    // Auth Check
    if (!auth.user) {
        elements.authModal.classList.remove('hidden');
        return;
    }

    try {
        switchView('processing');
        startTimer();

        const response = await api.startSessionUrl(url, state.llmConfig);
        state.sessionId = response.session_id;

        startStreaming(state.sessionId);
    } catch (error) {
        alert(error.message);
        switchView('start');
    }
}

async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) {
        addDebugLog('WARN', 'No file selected');
        return;
    }

    // Auth Check
    if (!auth.user) {
        elements.authModal.classList.remove('hidden');
        // Clear the input so the change event can fire again if they select the same file
        e.target.value = '';
        return;
    }

    try {
        addDebugLog('INFO', 'File upload started', { fileName: file.name, fileSize: file.size, fileType: file.type });
        switchView('processing');
        startTimer();

        addDebugLog('INFO', 'Uploading file to backend', { fileName: file.name });
        const response = await api.startSessionUpload(file, state.llmConfig);
        addDebugLog('INFO', 'Upload response received', response);
        state.sessionId = response.session_id;
        addDebugLog('INFO', 'Session ID assigned', { sessionId: state.sessionId });

        addDebugLog('INFO', 'Starting streaming for session', { sessionId: state.sessionId });
        startStreaming(state.sessionId);
    } catch (error) {
        addDebugLog('ERROR', 'Upload failed', { message: error.message, stack: error.stack });
        alert(error.message);
        switchView('start');
    }
}

// Streaming Logic
function startStreaming(sessionId) {
    addDebugLog('INFO', 'startStreaming called', { sessionId });
    if (state.eventSource) {
        try {
            state.eventSource.close();
        } catch (e) {
            addDebugLog('WARN', 'Failed to close previous EventSource', { message: e.message });
        }
    }

    const eventSource = api.getEventSource(sessionId);
    state.eventSource = eventSource;
    addDebugLog('INFO', 'EventSource created', { url: eventSource.url, readyState: eventSource.readyState });

    // Initialize canvas
    if (elements.canvasContent) {
        elements.canvasContent.innerHTML = '<p class="text-slate-500 italic">Starting analysis...</p>';
        addDebugLog('INFO', 'Canvas initialized');
    } else {
        addDebugLog('ERROR', 'Canvas content element not found');
    }
    lucide.createIcons();

    eventSource.onopen = () => {
        addDebugLog('INFO', 'EventSource connected (onopen)');
    };

    eventSource.onmessage = (event) => {
        addDebugLog('DEBUG', 'Message received from EventSource', { data: event.data.substring(0, 100) });
        try {
            const data = JSON.parse(event.data);
            addDebugLog('INFO', 'Message parsed', { type: data.type, agent: data.agent });

            if (data.type === 'agent_start') {
                updateCurrentAgent(data.agent);
                appendToCanvas(data.message);
                addLog(data.agent, data.message);
            } else if (data.type === 'agent_update') {
                updateCurrentAgent(data.agent);
                appendToCanvas(data.message);
                addLog(data.agent, data.message);
            } else if (data.type === 'start') {
                appendToCanvas(data.message);
                addLog('system', data.message);
            } else if (data.type === 'complete') {
                appendToCanvas(data.message);
                addLog('system', data.message);
                eventSource.close();
                addDebugLog('INFO', 'Stream complete, checking session status');
                checkCompletion();
            } else if (data.type === 'error') {
                appendToCanvas(data.message);
                addLog('system', data.message);
                eventSource.close();
                addDebugLog('ERROR', 'Stream error received, checking session status');
                checkCompletion();
            } else {
                addDebugLog('DEBUG', 'Other message type received', { type: data.type });
            }
        } catch (parseError) {
            addDebugLog('ERROR', 'Failed to parse message', { error: parseError.message });
        }
    };

    eventSource.onerror = (error) => {
        addDebugLog('ERROR', 'EventSource error', {
            readyState: eventSource.readyState,
            errorType: error.type,
            message: error.message
        });
        if (eventSource.readyState === EventSource.CLOSED) {
            eventSource.close();
            addDebugLog('INFO', 'EventSource closed, calling checkCompletion');
            checkCompletion();
        } else {
            addDebugLog('WARN', 'EventSource reconnecting; keeping stream open');
        }
    };
}

function updateCurrentAgent(agent) {
    if (elements.currentAgent) {
        const friendlyNames = {
            'paper_analysis': 'Paper Analysis',
            'web_research': 'Web Research',
            'sota_tracker': 'SOTA Tracker',
            'comparative_analysis': 'Comparative Analysis',
            'direction_advisor': 'Direction Advisor',
            'report_generation': 'Report Generation'
        };
        elements.currentAgent.innerHTML = `
            <span class="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            <span>${friendlyNames[agent] || agent}</span>
        `;
    }
}

function appendToCanvas(message) {
    if (elements.canvasContent) {
        // Append as a new paragraph, rendered as markdown
        const para = document.createElement('div');
        para.className = 'mb-2 animate-fade-in';
        para.innerHTML = ui.renderMarkdown(message);
        elements.canvasContent.appendChild(para);

        // Scroll to bottom
        if (elements.liveCanvas) {
            elements.liveCanvas.scrollTop = elements.liveCanvas.scrollHeight;
        }
    } else {
        console.error('❌ Canvas content element not found!');
    }
}

function addLog(agent, message) {
    elements.logsContainer.insertAdjacentHTML('afterbegin', ui.renderLogEntry(agent, message));
}

async function checkCompletion() {
    try {
        addDebugLog('INFO', 'checkCompletion called', { sessionId: state.sessionId });
        const session = await api.getSession(state.sessionId);
        addDebugLog('INFO', 'Session status retrieved', { status: session.status });

        if (session.status === 'completed') {
            stopTimer();
            addDebugLog('INFO', 'Session completed, loading report');
            loadReport(session.session_id);
        } else if (session.status === 'error') {
            stopTimer();
            addDebugLog('ERROR', 'Session error', { errors: session.errors });
            alert('Analysis failed: ' + JSON.stringify(session.errors));
            switchView('start');
        } else {
            // Still processing, keep the processing view visible and retry
            addDebugLog('INFO', 'Session still processing, will retry in 2s', { status: session.status });
            if (views.processing.classList.contains('hidden')) {
                addDebugLog('WARN', 'Processing view was hidden, showing it again');
                switchView('processing');
            }
            setTimeout(checkCompletion, 2000);
        }
    } catch (e) {
        addDebugLog('ERROR', 'checkCompletion error', { message: e.message, stack: e.stack });
        // Optionally, show an error or keep retrying
        setTimeout(checkCompletion, 2000);
    }
}

// Report Logic
async function loadReport(sessionId) {
    try {
        const report = await api.getReport(sessionId);
        state.reportData = report;
        renderReport(report);
        switchView('report');
    } catch (e) {
        alert('Failed to load report: ' + e.message);
    }
}

function renderReport(report) {
    elements.reportTitle.textContent = report.paper_title || 'Research Report';

    const tabs = [
        { id: 'executive_summary', label: 'Executive Summary' },
        { id: 'research_findings', label: 'Findings' },
        { id: 'sota_analysis', label: 'SOTA Analysis' },
        { id: 'comparative_analysis', label: 'Comparative' },
        { id: 'direction_advisor', label: 'Strategic Direction' },
        { id: 'future_directions', label: 'Future Work' }
    ];

    // Render Tabs
    elements.reportTabs.innerHTML = tabs.map(tab => `
        <button onclick="window.switchTab('${tab.id}')" 
            class="w-full text-left px-4 py-2 rounded-lg text-sm font-medium transition-colors ${state.activeTab === tab.id ? 'bg-blue-50 text-accent' : 'text-gray-600 hover:bg-gray-100'}"
            id="tab-${tab.id}">
            ${tab.label}
        </button>
    `).join('');

    // Render Content
    renderTabContent();
}

window.switchTab = (tabId) => {
    state.activeTab = tabId;

    // Update Tab Styles
    document.querySelectorAll('#report-tabs button').forEach(btn => {
        if (btn.id === `tab-${tabId}`) {
            btn.className = 'w-full text-left px-4 py-2 rounded-lg text-sm font-medium transition-colors bg-blue-50 text-accent';
        } else {
            btn.className = 'w-full text-left px-4 py-2 rounded-lg text-sm font-medium transition-colors text-gray-600 hover:bg-gray-100';
        }
    });

    renderTabContent();
};

function renderTabContent() {
    const data = state.reportData[state.activeTab];
    let content = '';

    if (typeof data === 'string') {
        content = ui.renderMarkdown(data);
    } else if (typeof data === 'object') {
        content = ui.renderMarkdown('```json\n' + JSON.stringify(data, null, 2) + '\n```');
    } else {
        content = '<p class="text-gray-500 italic">No data available for this section.</p>';
    }

    elements.reportContent.innerHTML = content;
}

// Chat Logic
function openChat() {
    elements.chatDrawer.classList.remove('translate-x-full');
    state.currentDeepDiveField = state.activeTab;
    elements.chatContextName.textContent = state.activeTab.replace('_', ' ').toUpperCase();
}

function closeChat() {
    elements.chatDrawer.classList.add('translate-x-full');
}

async function handleChatMessage(e) {
    e.preventDefault();
    const message = elements.chatInput.value.trim();
    if (!message) return;

    // Add User Message
    elements.chatMessages.insertAdjacentHTML('beforeend', ui.renderChatMessage('user', message));
    elements.chatInput.value = '';
    scrollToBottom();

    try {
        // Add Loading
        const loadingId = 'loading-' + Date.now();
        elements.chatMessages.insertAdjacentHTML('beforeend', `
            <div id="${loadingId}" class="flex justify-start animate-fade-in">
                <div class="bg-gray-100 p-3 rounded-2xl rounded-bl-none">
                    <div class="loader"></div>
                </div>
            </div>
        `);
        scrollToBottom();

        const response = await api.sendChatMessage(state.sessionId, state.currentDeepDiveField, message);

        // Remove Loading
        document.getElementById(loadingId).remove();

        // Add Assistant Message
        elements.chatMessages.insertAdjacentHTML('beforeend', ui.renderChatMessage('assistant', response.answer));
        scrollToBottom();

    } catch (error) {
        alert('Chat failed: ' + error.message);
    }
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Export Logic
function handleExportReport() {

    try {
        if (!state.reportData) {
            alert('❌ No report to export');
            return;
        }

        const reportTitle = state.reportData.paper_title || 'Research Report';
        const fullMarkdown = generateFullReportMarkdown();

        if (!fullMarkdown) {
            alert('❌ Report content is empty');
            return;
        }

        // Show download options
        const format = prompt('Export format:\n(md = Markdown)\n(html = HTML)\n(txt = Text)\n\nEnter: md, html, or txt', 'md');

        if (!format) return; // User cancelled

        const formatLower = format.toLowerCase().trim();

        if (formatLower === 'md' || formatLower === 'markdown') {
            exportAsMarkdown(fullMarkdown, reportTitle);
        } else if (formatLower === 'html') {
            exportAsHTML(fullMarkdown, reportTitle);
        } else if (formatLower === 'txt' || formatLower === 'text') {
            exportAsText(fullMarkdown, reportTitle);
        } else {
            alert('❌ Invalid format. Use: md, html, or txt');
        }

    } catch (error) {
        console.error('❌ Export error:', error);
        alert('❌ Export failed: ' + error.message);
    }
}

function generateFullReportMarkdown() {
    if (!state.reportData) return '';

    const sections = [
        { id: 'executive_summary', title: 'Executive Summary' },
        { id: 'research_findings', title: 'Research Findings' },
        { id: 'sota_analysis', title: 'SOTA Analysis' },
        { id: 'comparative_analysis', title: 'Comparative Analysis' },
        { id: 'direction_advisor', title: 'Strategic Direction' },
        { id: 'future_directions', title: 'Future Work' }
    ];

    let fullMarkdown = `# ${state.reportData.paper_title || 'Research Report'}\n\n`;

    // Add Authors if available
    if (state.reportData.authors && state.reportData.authors.length > 0) {
        fullMarkdown += `**Authors:** ${state.reportData.authors.join(', ')}\n\n`;
    }

    sections.forEach(section => {
        const content = state.reportData[section.id];
        if (content) {
            fullMarkdown += `## ${section.title}\n\n`;
            if (typeof content === 'string') {
                fullMarkdown += content + '\n\n';
            } else {
                fullMarkdown += '```json\n' + JSON.stringify(content, null, 2) + '\n```\n\n';
            }
            fullMarkdown += '---\n\n';
        }
    });

    return fullMarkdown;
}

function exportAsMarkdown(html, title) {
    try {

        // Simple HTML to Markdown conversion
        let markdown = `# ${title}\n\n`;

        // Convert headings
        markdown += html
            .replace(/<h1[^>]*>([^<]*)<\/h1>/g, '# $1\n\n')
            .replace(/<h2[^>]*>([^<]*)<\/h2>/g, '## $1\n\n')
            .replace(/<h3[^>]*>([^<]*)<\/h3>/g, '### $1\n\n')
            .replace(/<h4[^>]*>([^<]*)<\/h4>/g, '#### $1\n\n')
            .replace(/<h5[^>]*>([^<]*)<\/h5>/g, '##### $1\n\n')
            .replace(/<h6[^>]*>([^<]*)<\/h6>/g, '###### $1\n\n')
            // Convert paragraphs
            .replace(/<p[^>]*>([^<]*)<\/p>/g, '$1\n\n')
            // Convert bold
            .replace(/<strong[^>]*>([^<]*)<\/strong>/g, '**$1**')
            .replace(/<b[^>]*>([^<]*)<\/b>/g, '**$1**')
            // Convert italic
            .replace(/<em[^>]*>([^<]*)<\/em>/g, '*$1*')
            .replace(/<i[^>]*>([^<]*)<\/i>/g, '*$1*')
            // Convert lists
            .replace(/<li[^>]*>([^<]*)<\/li>/g, '- $1\n')
            .replace(/<ul[^>]*>(.*?)<\/ul>/gs, '$1\n')
            .replace(/<ol[^>]*>(.*?)<\/ol>/gs, '$1\n')
            // Convert line breaks
            .replace(/<br\s*\/?>/g, '\n')
            // Remove other HTML tags
            .replace(/<[^>]+>/g, '')
            // Clean up whitespace
            .replace(/\n{3,}/g, '\n\n');

        downloadFile(markdown, `${title.replace(/\s+/g, '-')}.md`, 'text/markdown');
        alert('✅ Report exported as Markdown');

    } catch (error) {
        console.error('❌ Markdown export error:', error);
        alert('❌ Markdown export failed: ' + error.message);
    }
}

function exportAsHTML(html, title) {
    try {

        const fullHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #fff;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 600;
        }
        h1 { font-size: 2em; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.25em; }
        p { margin-bottom: 1em; }
        code {
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }
        pre {
            background: #1f2937;
            color: #f3f4f6;
            padding: 1em;
            border-radius: 6px;
            overflow-x: auto;
            margin: 1em 0;
            font-family: 'Monaco', 'Courier New', monospace;
        }
        pre code { background: none; padding: 0; color: inherit; }
        ul, ol { margin: 1em 0 1em 2em; }
        li { margin-bottom: 0.5em; }
        blockquote {
            border-left: 4px solid #3b82f6;
            padding-left: 1em;
            margin: 1em 0;
            color: #666;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 0.75em;
            text-align: left;
        }
        th { background: #f3f4f6; font-weight: 600; }
        a { color: #3b82f6; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .timestamp {
            text-align: center;
            color: #999;
            font-size: 0.9em;
            margin-top: 3em;
            border-top: 1px solid #ddd;
            padding-top: 1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>${title}</h1>
        ${html}
        <div class="timestamp">
            <p>Exported on ${new Date().toLocaleString()}</p>
            <p>Generated by Research Copilot</p>
        </div>
    </div>
</body>
</html>`;

        downloadFile(fullHTML, `${title.replace(/\s+/g, '-')}.html`, 'text/html');
        console.log('✅ HTML export successful');
        alert('✅ Report exported as HTML');

    } catch (error) {
        console.error('❌ HTML export error:', error);
        alert('❌ HTML export failed: ' + error.message);
    }
}

function exportAsText(html, title) {
    try {
        console.log('📝 Exporting as Text...');

        // Strip HTML tags and convert to plain text
        let text = html
            .replace(/<br\s*\/?>/g, '\n')
            .replace(/<[^>]+>/g, '')
            .replace(/&nbsp;/g, ' ')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&amp;/g, '&')
            .replace(/&quot;/g, '"')
            .replace(/&apos;/g, "'");

        // Add title and timestamp
        const fullText = `${title}\n${'='.repeat(title.length)}\n\nExported on ${new Date().toLocaleString()}\n\n${text}`;

        downloadFile(fullText, `${title.replace(/\s+/g, '-')}.txt`, 'text/plain');
        console.log('✅ Text export successful');
        alert('✅ Report exported as Text');

    } catch (error) {
        console.error('❌ Text export error:', error);
        alert('❌ Text export failed: ' + error.message);
    }
}

function downloadFile(content, filename, mimeType) {
    try {
        console.log(`📥 Starting download: ${filename}`);

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');

        link.href = url;
        link.download = filename;
        link.style.display = 'none';

        // Append to body, click, and remove
        document.body.appendChild(link);
        link.click();

        // Clean up
        setTimeout(() => {
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            console.log(`✅ Download complete: ${filename}`);
        }, 100);

    } catch (error) {
        console.error('❌ Download error:', error);
        throw error;
    }
}

// Helpers
function switchView(viewId) {
    console.log(`📺 Switching view to: ${viewId}`);
    Object.values(views).forEach(el => el.classList.add('hidden'));
    views[viewId].classList.remove('hidden');
}

let timerInterval;
function startTimer() {
    console.log('⏱️ Timer started');
    let seconds = 0;
    elements.timer.textContent = "00:00";
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
        seconds++;
        const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
        const secs = (seconds % 60).toString().padStart(2, '0');
        elements.timer.textContent = `${mins}:${secs}`;
    }, 1000);
}

function stopTimer() {
    console.log('⏹️ Timer stopped');
    clearInterval(timerInterval);
}

function resetSession() {
    state.sessionId = null;
    state.status = 'idle';
    state.reportData = null;
    state.chatHistory = [];
    elements.logsContainer.innerHTML = '';
    elements.currentAgent.innerHTML = '<span class="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span><span>Waiting...</span>';
    elements.chatMessages.innerHTML = '';
    switchView('start');
}

// Run
document.addEventListener('DOMContentLoaded', init);
