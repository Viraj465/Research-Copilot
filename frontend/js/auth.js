
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'
import config from './config.js';

const SUPABASE_URL = config.SUPABASE_URL;

const SUPABASE_ANON_KEY = config.SUPABASE_ANON_KEY;

let supabase = null;

try {
    if (SUPABASE_URL && SUPABASE_URL !== 'YOUR_SUPABASE_URL' && SUPABASE_URL.startsWith('http')) {
        supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    } else {
        console.warn('Supabase credentials not set. Auth will be disabled.');
    }
} catch (e) {
    console.error('Failed to initialize Supabase client:', e);
}

export { supabase };

export const auth = {
    user: null,
    token: null,

    async init() {
        if (!supabase) return;

        try {
            // Check active session
            const { data: { session } } = await supabase.auth.getSession();
            this.updateState(session);

            // Listen for changes
            supabase.auth.onAuthStateChange((_event, session) => {
                this.updateState(session);
            });
        } catch (e) {
            console.error('Auth init error:', e);
        }
    },

    updateState(session) {
        this.user = session?.user || null;
        this.token = session?.access_token || null;

        // Dispatch event for UI updates
        window.dispatchEvent(new CustomEvent('auth:change', {
            detail: { user: this.user }
        }));
    },

    async signInWithGoogle() {
        if (!supabase) {
            alert('Please configure Supabase credentials in js/auth.js');
            return;
        }
        const { data, error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
    redirectTo: window.location.href 
}
        });
        if (error) throw error;
        return data;
    },

    async signInWithGithub() {
        if (!supabase) {
            alert('Please configure Supabase credentials in js/auth.js');
            return;
        }
        const { data, error } = await supabase.auth.signInWithOAuth({
            provider: 'github',
            options: {
                redirectTo: window.location.origin
            }
        });
        if (error) throw error;
        return data;
    },

    async signOut() {
        if (!supabase) return;
        const { error } = await supabase.auth.signOut();
        if (error) throw error;
    },

    getToken() {
        return this.token;
    }
};
