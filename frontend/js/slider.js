export function initSlider() {
    const slider = document.getElementById('history-slider');
    const valueDisplay = document.getElementById('history-value');

    if (!slider || !valueDisplay) return;

    // Initialize from local storage or default
    const savedValue = localStorage.getItem('historyDepth') || '0';
    slider.value = savedValue;
    valueDisplay.textContent = savedValue;

    slider.addEventListener('input', (e) => {
        const val = e.target.value;
        valueDisplay.textContent = val;
        localStorage.setItem('historyDepth', val);
    });
}

export function getHistoryDepth() {
    return parseInt(localStorage.getItem('historyDepth') || '0', 10);
}
