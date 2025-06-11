//JavaScript Functions
function setIfExist(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value || "--";
}

async function updateModelStats(modelName) {
    try {
        const res = await fetch(`/model-info?model=${modelName}`);
        const data = await res.json();

        if (data.error) {
            console.error("Erreur stats modèle:", data.error);
            return;
        }

        setIfExist("stat-train", data.train_accuracy);
        setIfExist("stat-val", data.val_accuracy);
        setIfExist("stat-time", data.training_time);

        setIfExist("stat-test-accuracy", data.test_accuracy);
        setIfExist("stat-precision", data.precision);
        setIfExist("stat-recall", data.recall);
        setIfExist("stat-f1", data.f1_score);
        
    } catch (err) {
        console.error("Erreur récupération stats:", err);
    }
}


function initModelSelector() {
    const modelSelector = document.getElementById("model-selector");
    if (!modelSelector) return;

    modelSelector.addEventListener("change", function () {
        const selected = this.value;
        updateModelStats(selected);
    });

    const initialModel = modelSelector.value;
    updateModelStats(initialModel);
}


//SQL Function
async function analyzeSQL() {
    const input = document.getElementById('query').value.trim();
    const resultContainer = document.getElementById('result-container');
    const predictBtn = document.getElementById('sql-predict-btn');
    const selectedModel = document.getElementById('model-selector').value;

    document.querySelectorAll('.result-card').forEach(el => {
        el.style.display = 'none';
    });
    resultContainer.style.display = 'none';
    
    if (!input) {
        alert('❌ Veuillez entrer une requête SQL');
        return;
    }
    try {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Analyse...';
        
        const response = await fetch('/predict-sql', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: input, model: selectedModel })
        });
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        // Show results
        resultContainer.style.display = 'block';
        document.getElementById(data.result === 1 ? 'result-danger' : 'result-safe').style.display = 'flex';
        addToHistory(input, data.result, 'sql');
        
    } catch (error) {
        console.error('Erreur:', error);
        alert(`❌ ${error.message || 'Erreur lors de l\'analyse'}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="bx bx-search-alt"></i> Analyser';
    }
}

async function analyzeHttp() {
    const input = document.getElementById('query').value.trim();
    const resultContainer = document.getElementById('result-container');
    const predictBtn = document.getElementById('http-predict-btn');

    document.querySelectorAll('.result-card').forEach(el => {
        el.style.display = 'none';
    });
    resultContainer.style.display = 'none';
    if (!input) {
        alert('❌ Veuillez entrer une requête HTTP');
        return;
    }
    try {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Analyse...';
        const response = await fetch('/predict-http', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: input })
        });
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        // Show results
        resultContainer.style.display = 'block';
        document.getElementById(data.result === 1 ? 'result-danger' : 'result-safe').style.display = 'flex';
        addToHistory(input, data.result, 'http');

    } catch (error) {
        console.error('Erreur:', error);
        alert(`❌ ${error.message || 'Erreur lors de l\'analyse'}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="bx bx-search-alt"></i> Analyser';
    }
}

// Helper function to escape HTML
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Main
document.addEventListener('DOMContentLoaded', () => {

    initSidebar();
    
    initModelSelector();

    // Gestion des boutons d'analyse
    document.getElementById('sql-predict-btn')?.addEventListener('click', () => analyzeSQL());
    document.getElementById('http-predict-btn')?.addEventListener('click', () => analyzeHttp());
    
    //charts
    if (window.location.pathname.includes('/analystic')) initCharts();

    
});

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//sideBar
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

function addToHistory(query, result, type = 'sql') {
    const historyTable = document.querySelector('.history-table tbody');
    if (!historyTable) return;

    // Trouver le modèle sélectionné
    const selectedModel = document.getElementById('model-selector')?.value || 'svm';
    // Générer un label de modèle friendly
    let modelLabel = '';
    let modelClass = '';
    switch(selectedModel) {
        case 'svm':
            modelLabel = 'SVM';
            modelClass = 'svm';
            break;
        case 'randomforest':
            modelLabel = 'Random Forest';
            modelClass = 'rf';
            break;
        case 'logisticregression':
            modelLabel = 'Logistic Regression';
            modelClass = 'lr';
            break;
        default:
            modelLabel = selectedModel;
            modelClass = '';
    }

    // Générer un nouvel ID (ex: #003)
    let newId = 1;
    const rows = historyTable.querySelectorAll('tr');
    if (rows.length > 0) {
        const ids = Array.from(rows).map(row => {
            const td = row.querySelector('td:nth-child(2)');
            if (td) {
                const num = td.textContent.replace('#','');
                return parseInt(num, 10) || 0;
            }
            return 0;
        });
        newId = Math.max(...ids) + 1;
    }
    const idString = '#' + newId.toString().padStart(3, '0');

    // Affichage résultat
    const isDanger = result === 1;
    const resultBadge = isDanger
        ? `<span class="status-badge danger"><i class='bx bx-error-circle'></i> Menace</span>`
        : `<span class="status-badge safe"><i class='bx bx-check-circle'></i> Sécurisée</span>`;

    // Actions boutons
    const actionButtons = `
        <div class="action-buttons">
            <button class="btn-icon" title="Voir détails"><i class='bx bx-show'></i></button>
            <button class="btn-icon" title="Réanalyser"><i class='bx bx-refresh'></i></button>
            <button class="btn-icon danger" title="Supprimer"><i class='bx bx-trash'></i></button>
        </div>
    `;

    // Date formatée
    const now = new Date();
    const dateString = now.toLocaleDateString('fr-FR') + ' ' + now.toLocaleTimeString('fr-FR', {hour: '2-digit', minute: '2-digit'});

    // Ajoute la ligne en haut du tableau
    const newRow = document.createElement('tr');
    newRow.innerHTML = `
        <td><input type="checkbox"></td>
        <td>${idString}</td>
        <td class="query-cell">
            <div class="query-preview"><code>${escapeHtml(query)}</code></div>
        </td>
        <td>${resultBadge}</td>
        <td><span class="model-badge ${modelClass}">${modelLabel}</span></td>
        <td><time>${dateString}</time></td>
        <td>${actionButtons}</td>
    `;
    historyTable.prepend(newRow);
}

function bientot(fileId) {
    alert('Fonctionnalité bientôt disponible !');
}

