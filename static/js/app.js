document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked tab and its content
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });

    // Single comment analysis
    const analyzeBtn = document.getElementById('analyze-btn');
    const singleCommentInput = document.getElementById('single-comment');
    const singleResultArea = document.getElementById('single-result');

    analyzeBtn.addEventListener('click', async () => {
        const text = singleCommentInput.value.trim();
        if (!text) {
            alert('Пожалуйста, введите комментарий для анализа.');
            return;
        }

        // Show loading
        singleResultArea.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const result = await response.json();
            displaySingleResult(result);
        } catch (error) {
            singleResultArea.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
    });

    // Batch comments analysis
    const analyzeBatchBtn = document.getElementById('analyze-batch-btn');
    const batchCommentsInput = document.getElementById('batch-comments');
    const batchResultsArea = document.getElementById('batch-results');

    analyzeBatchBtn.addEventListener('click', async () => {
        const text = batchCommentsInput.value.trim();
        if (!text) {
            alert('Пожалуйста, введите комментарии для анализа.');
            return;
        }

        // Split text by newlines to get individual comments
        const texts = text.split('\n').filter(line => line.trim());
        if (texts.length === 0) {
            alert('Пожалуйста, введите хотя бы один комментарий для анализа.');
            return;
        }

        // Show loading
        batchResultsArea.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

        try {
            const response = await fetch('/api/classify_batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ texts })
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const data = await response.json();
            displayBatchResults(data.results);
        } catch (error) {
            batchResultsArea.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
    });

    // Display functions
    function displaySingleResult(result) {
        const html = createResultItemHtml(result);
        singleResultArea.innerHTML = html;
    }

    function displayBatchResults(results) {
        if (!results || results.length === 0) {
            batchResultsArea.innerHTML = '<div class="result-placeholder">Результаты не найдены</div>';
            return;
        }

        const html = results.map(result => createResultItemHtml(result)).join('');
        batchResultsArea.innerHTML = html;
    }

    function createResultItemHtml(result) {
        const text = result.text;
        const labels = result.predicted_labels;
        const probabilities = result.probabilities;

        const labelsHtml = labels.map(label => {
            // Перевод меток на русский
            let russianLabel = label;
            if (label === "NORMAL") russianLabel = "НОРМАЛЬНО";
            if (label === "INSULT") russianLabel = "ОСКОРБЛЕНИЕ";
            if (label === "THREAT") russianLabel = "УГРОЗА";
            if (label === "OBSCENITY") russianLabel = "НЕПРИСТОЙНОСТЬ";

            return `<span class="label ${label.toLowerCase()}">${russianLabel}</span>`;
        }).join('');

        const probabilitiesHtml = Object.entries(probabilities).map(([label, value]) => {
            // Перевод меток на русский
        let russianLabel = label;
        if (label === "NORMAL") russianLabel = "НОРМАЛЬНО";
        if (label === "INSULT") russianLabel = "ОСКОРБЛЕНИЕ";
        if (label === "THREAT") russianLabel = "УГРОЗА";
        if (label === "OBSCENITY") russianLabel = "НЕПРИСТОЙНОСТЬ";

        return `
                <div class="probability">
                    <div class="probability-label">${russianLabel}</div>
                    <div class="probability-value">${(value * 100).toFixed(1)}%</div>
                </div>
            `;
        }).join('');

        return `
            <div class="result-item">
                <div class="result-text">${escapeHtml(text)}</div>
                <div class="labels">${labelsHtml}</div>
                <h4>Оценка уверенности:</h4>
                <div class="probabilities">${probabilitiesHtml}</div>
            </div>
        `;
    }

    // Helper function to escape HTML
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});