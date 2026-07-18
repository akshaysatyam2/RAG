import { initTheme } from './theme.js';
import { initSlider } from './slider.js';
import { initDocuments } from './documents.js';
import { initChat } from './chat.js';

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSlider();
    initDocuments();
    initChat();
    initMobileMenu();
});

function initMobileMenu() {
    const toggleBtn = document.getElementById('mobile-menu-toggle');
    const sidebar = document.getElementById('sidebar');
    
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
        
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 && 
                !sidebar.contains(e.target) && 
                !toggleBtn.contains(e.target) && 
                sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
            }
        });
    }
}

export function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        if (container.contains(toast)) {
            container.removeChild(toast);
        }
    }, 3500);
}
