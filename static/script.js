function initSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const toggle = document.querySelector('.toggle');
    
    const savedState = localStorage.getItem('sidebarState');
    
    if (savedState === 'closed') {
        sidebar.classList.add('close');
    } else {
        sidebar.classList.remove('close');
        if (savedState === null) {
            localStorage.setItem('sidebarState', 'open');
        }
    }
    
    if (toggle) {
        toggle.addEventListener('click', () => {
            sidebar.classList.toggle('close');
            localStorage.setItem('sidebarState', 
                sidebar.classList.contains('close') ? 'closed' : 'open');
        });
    }
}

async function analyzeQuery(type) {
    const input = document.getElementById('query').value.trim();
    const resultContainer = document.getElementById('result-container');
    const predictBtn = document.getElementById(`${type}-predict-btn`);
    
    // Masquer les résultats précédents
    document.querySelectorAll('.sql-result').forEach(el => {
        el.style.display = 'none';
    });
    resultContainer.style.display = 'none';
    
    if (!input) {
        alert(`❌ Veuillez entrer une requête ${type.toUpperCase()}`);
        return;
    }
    
    try {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Analyse...';
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: input, 
                type: type 
            })
        });
        
        const data = await response.json();
        
        resultContainer.style.display = 'block';
        document.getElementById(data.result === 1 ? 'id-y-attack' : 'id-no-attack').style.display = 'flex';
        
        addToHistory(input, data.result);
        
    } catch (error) {
        console.error('Erreur:', error);
        alert('❌ Erreur lors de l\'analyse');
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="bx bx-search-alt"></i> Analyser';
    }
}

function addToHistory(query, result) {
    const historyTable = document.querySelector('.history-table tbody');
    if (!historyTable) return;
    
    const newRow = document.createElement('tr');
    newRow.innerHTML = `
        <td>${historyTable.children.length + 1}</td>
        <td class="query-preview" title="${query}">${query}</td>
        <td><span class="badge ${result === 1 ? 'danger' : 'safe'}">
            ${result === 1 ? 'Danger' : 'Sûr'}
        </span></td>
        <td>${new Date().toLocaleString()}</td>
    `;
    
    historyTable.prepend(newRow);
}
// Initialisation des graphiques
function initCharts() {
    // Graphique 1 : Répartition des types d'attaques
    const ctx1 = document.getElementById('attacksByTypeChart').getContext('2d');
    new Chart(ctx1, {
        type: 'doughnut',
        data: {
            labels: ['SQL Injection', 'XSS', 'Path Traversal', 'LFI', 'Autres'],
            datasets: [{
                data: [65, 15, 10, 5, 5],
                backgroundColor: [
                    '#4361ee',
                    '#3f37c9',
                    '#4cc9f0',
                    '#4895ef',
                    '#f72585'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });

    // Graphique 2 : Précision sur 30 jours
    const ctx2 = document.getElementById('accuracyOverTimeChart').getContext('2d');
    new Chart(ctx2, {
        type: 'line',
        data: {
            labels: Array.from({length: 30}, (_, i) => `J-${30-i}`),
            datasets: [{
                label: 'Précision SVC',
                data: Array.from({length: 30}, () => Math.random() * 10 + 85),
                borderColor: '#4361ee',
                tension: 0.3,
                fill: false
            }, {
                label: 'Précision MultinomialNB',
                data: Array.from({length: 30}, () => Math.random() * 10 + 65),
                borderColor: '#3f37c9',
                tension: 0.3,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    min: 50,
                    max: 100
                }
            }
        }
    });
}





// Main
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    
    // Gestion des boutons d'analyse
    document.getElementById('sql-predict-btn')?.addEventListener('click', () => analyzeQuery('sql'));
    document.getElementById('http-predict-btn')?.addEventListener('click', () => analyzeQuery('http'));
    
    // Entrée déclenche l'analyse
    document.getElementById('query')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const path = window.location.pathname;
            if (path.includes('/http')) {
                analyzeQuery('http');
            } else if (path.includes('/sql')) {
                analyzeQuery('sql');
            }
        }
    });

    //charts
    if (window.location.pathname.includes('/analystic')) {
        initCharts();
    }
});

function bientot(fileId) {
    alert('Fonctionnalité d\'analyse bientôt disponible !');
}