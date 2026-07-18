import { api } from './api.js';
import { showToast } from './app.js';
import { getHistoryDepth } from './slider.js';

let chatHistory = [];

export function initChat() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const promptBtns = document.querySelectorAll('.prompt-btn');

    if (!input || !sendBtn) return;

    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 150) + 'px';
        sendBtn.disabled = input.value.trim() === '';
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) sendBtn.click();
        }
    });

    sendBtn.addEventListener('click', () => {
        const query = input.value.trim();
        if (query) {
            input.value = '';
            input.style.height = 'auto';
            sendBtn.disabled = true;
            handleQuery(query);
        }
    });

    promptBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            handleQuery(btn.textContent);
        });
    });
}

async function handleQuery(query) {
    const container = document.getElementById('chat-messages');
    
    const emptyState = container.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    appendMessage('user', query);
    
    const depth = getHistoryDepth();
    const truncatedHistory = chatHistory.slice(-depth || undefined);
    if (depth === 0) chatHistory = []; 

    const typingId = `typing-${Date.now()}`;
    const typingHtml = `
        <div class="message bot" id="${typingId}">
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </div>
            <div class="message-content">
                <div class="message-bubble typing-indicator">
                    <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                </div>
            </div>
        </div>
    `;
    container.insertAdjacentHTML('beforeend', typingHtml);
    scrollToBottom();

    try {
        const response = await api.sendQuery(query, truncatedHistory, 5);
        
        document.getElementById(typingId)?.remove();

        appendMessage('bot', response.answer, response.sources);
        
        chatHistory.push({ role: 'user', content: query });
        chatHistory.push({ role: 'assistant', content: response.answer });

    } catch (error) {
        document.getElementById(typingId)?.remove();
        showToast(error.message, 'error');
        appendMessage('bot', 'Sorry, I encountered an error while processing your request.');
    }
}

function appendMessage(role, content, sources = []) {
    const container = document.getElementById('chat-messages');
    const avatar = role === 'user' 
        ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
        : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';

    const htmlContent = role === 'bot' && window.marked ? window.marked.parse(content) : escapeHtml(content);

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources-container">
                ${sources.map((s, i) => `
                    <details class="source-card">
                        <summary>Source ${i + 1}: ${escapeHtml(s.title || 'Document')}</summary>
                        <div class="source-content">${escapeHtml(s.snippet || s.content || '')}</div>
                    </details>
                `).join('')}
            </div>
        `;
    }

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const msgHtml = `
        <div class="message ${role}">
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble">${htmlContent}</div>
                ${sourcesHtml}
                <div class="message-meta">
                    <span>${time}</span>
                    ${role === 'bot' ? `<button class="action-btn copy-btn" title="Copy response">Copy</button>` : ''}
                </div>
            </div>
        </div>
    `;

    container.insertAdjacentHTML('beforeend', msgHtml);
    
    if (role === 'bot') {
        const newMsg = container.lastElementChild;
        const copyBtn = newMsg.querySelector('.copy-btn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(content);
                showToast('Copied to clipboard');
            });
        }
    }

    scrollToBottom();
}

function scrollToBottom() {
    const container = document.getElementById('chat-messages');
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
