const config = {
    // TODO: Paste your Supabase URL and Anon Key here from your .env file
    SUPABASE_URL: "https://nrzbagnazxvxtjftdpvn.supabase.co",
    SUPABASE_ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yemJhZ25henh2eHRqZnRkcHZuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgwMzI0NzgsImV4cCI6MjA4MzYwODQ3OH0.Iq8g48uvvrGXrs-VvsgeWr35qqi_JfeOXF9SC8sY8mY',

    // API Keys for Research Copilot
    TAVILY_API_KEY: '',
    GROQ_API_KEY: '',
    GOOGLE_API_KEY: '',
    OPENAI_API_KEY: ''
};

// Load API keys from localStorage on startup
export function loadApiKeysFromStorage() {
    config.TAVILY_API_KEY = localStorage.getItem('tavily_api_key') || '';
    config.GROQ_API_KEY = localStorage.getItem('groq_api_key') || '';
    config.GOOGLE_API_KEY = localStorage.getItem('google_api_key') || '';
    config.OPENAI_API_KEY = localStorage.getItem('openai_api_key') || '';
}

// Save API keys to localStorage
export function saveApiKeysToStorage(tavilyKey, groqKey, googleKey, openaiKey) {
    if (tavilyKey) {
        localStorage.setItem('tavily_api_key', tavilyKey);
        config.TAVILY_API_KEY = tavilyKey;
    }
    if (groqKey) {
        localStorage.setItem('groq_api_key', groqKey);
        config.GROQ_API_KEY = groqKey;
    }
    if (googleKey) {
        localStorage.setItem('google_api_key', googleKey);
        config.GOOGLE_API_KEY = googleKey;
    }
    if (openaiKey) {
        localStorage.setItem('openai_api_key', openaiKey);
        config.OPENAI_API_KEY = openaiKey;
    }
}

// Get current API keys
export function getApiKeys() {
    return {
        tavily: config.TAVILY_API_KEY,
        groq: config.GROQ_API_KEY,
        google: config.GOOGLE_API_KEY,
        openai: config.OPENAI_API_KEY
    };
}

// Check if all required API keys are set
export function areApiKeysConfigured() {
    return !!(config.TAVILY_API_KEY && config.GROQ_API_KEY && config.GOOGLE_API_KEY);
}

export default config;
