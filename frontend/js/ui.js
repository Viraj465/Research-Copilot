export const ui = {
    /**
     * Render an agent status card
     */
    renderAgentCard(agentName, status, message) {
        const statusColors = {
            'running': 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-500',
            'completed': 'border-green-500 bg-green-50 dark:bg-green-900/20 dark:border-green-500',
            'error': 'border-red-500 bg-red-50 dark:bg-red-900/20 dark:border-red-500',
            'pending': 'border-gray-200 bg-white dark:bg-slate-800 dark:border-slate-700'
        };

        const icons = {
            'paper_analysis': 'file-text',
            'web_research': 'globe',
            'sota_tracker': 'trophy',
            'comparative_analysis': 'scale',
            'direction_advisor': 'compass',
            'report_generation': 'file-check'
        };

        const friendlyNames = {
            'paper_analysis': 'Paper Analysis',
            'web_research': 'Web Research',
            'sota_tracker': 'SOTA Tracker',
            'comparative_analysis': 'Comparative Analysis',
            'direction_advisor': 'Direction Advisor',
            'report_generation': 'Report Generation'
        };

        const colorClass = statusColors[status] || statusColors['pending'];
        const iconName = icons[agentName] || 'activity';
        const name = friendlyNames[agentName] || agentName;

        return `
            <div class="p-4 rounded-lg border ${colorClass} transition-all duration-300 animate-fade-in">
                <div class="flex items-center gap-3 mb-2">
                    <div class="p-2 rounded-full bg-white dark:bg-slate-900 shadow-sm">
                        <i data-lucide="${iconName}" class="w-5 h-5 text-slate-700 dark:text-slate-300"></i>
                    </div>
                    <h4 class="font-semibold text-sm text-slate-900 dark:text-white">${name}</h4>
                </div>
                <p class="text-xs text-slate-600 dark:text-slate-400 line-clamp-2">${message || 'Waiting...'}</p>
            </div>
        `;
    },

    /**
     * Render a log entry
     */
    renderLogEntry(agent, message) {
        const timestamp = new Date().toLocaleTimeString();
        return `
            <div class="hover:bg-slate-800/50 p-1 rounded">
                <span class="text-slate-500">[${timestamp}]</span>
                <span class="text-blue-400 font-bold">[${agent}]</span>
                <span class="text-slate-300">${message}</span>
            </div>
        `;
    },

    /**
     * Render markdown content safely
     */
    renderMarkdown(content) {
        const rawHtml = marked.parse(content);
        return DOMPurify.sanitize(rawHtml);
    },

    /**
     * Render chat message
     */
    renderChatMessage(role, content) {
        const isUser = role === 'user';
        const alignClass = isUser ? 'justify-end' : 'justify-start';
        const bgClass = isUser ? 'bg-accent text-white' : 'bg-gray-100 dark:bg-slate-800 text-gray-800 dark:text-gray-200';
        const roundedClass = isUser ? 'rounded-br-none' : 'rounded-bl-none';

        return `
            <div class="flex ${alignClass} animate-fade-in">
                <div class="max-w-[85%] p-3 rounded-2xl ${roundedClass} ${bgClass} text-sm shadow-sm">
                    ${isUser ? content : this.renderMarkdown(content)}
                </div>
            </div>
        `;
    }
};
