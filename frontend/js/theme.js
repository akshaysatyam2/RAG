export function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');
    
    document.documentElement.setAttribute('data-theme', initialTheme);
    
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }

    // Shortcut Ctrl+/
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            toggleTheme();
        }
    });
}

export function toggleTheme() {
    const root = document.documentElement;
    const currentTheme = root.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    root.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
}
