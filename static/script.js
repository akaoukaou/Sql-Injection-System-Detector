// Helper function to set the content if the element exists
function setIfExist(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value || "--";
}

// Fonction pour mettre à jour les statistiques des modèles (SQL, HTTP, ...)
async function updateModelStats(modelName, type) {
    try {
        const res = await fetch(`/model-info?model=${modelName}&type=${type}`);
        const data = await res.json();

        if (data.error) {
            console.error("Erreur stats modèle:", data.error);
            return;
        }

        setIfExist(`${type}-stat-train`, data.train_accuracy);
        setIfExist(`${type}-stat-val`, data.val_accuracy);
        setIfExist(`${type}-stat-time`, data.training_time);

        setIfExist(`${type}-stat-test-accuracy`, data.test_accuracy);
        setIfExist(`${type}-stat-precision`, data.precision);
        setIfExist(`${type}-stat-recall`, data.recall);
        setIfExist(`${type}-stat-f1`, data.f1_score);

    } catch (err) {
        console.error("Erreur récupération stats:", err);
    }
}

// Fonction pour initialiser les sélecteurs de modèles (SQL, HTTP, ...)
function initModelSelector(type) {
    const modelSelector = document.getElementById(`${type}-model-selector`);
    if (!modelSelector) return;

    modelSelector.addEventListener("change", function () {
        const selected = this.value;
        updateModelStats(selected, type);  // Passer le type dynamique (http ou sql)
    });

    const initialModel = modelSelector.value;
    updateModelStats(initialModel, type);  // Initialiser avec le modèle par défaut
}

// Fonction générique pour afficher le résultat (Danger ou Safe)
function displayResult(type, result) {
    const resultContainer = document.getElementById(`${type}-result-container`);
    const resultDanger = document.getElementById(`${type}-result-danger`);
    const resultSafe = document.getElementById(`${type}-result-safe`);

    resultDanger.style.display = 'none';
    resultSafe.style.display = 'none';
    resultContainer.style.display = 'none';

    if (result === 1) {
        resultDanger.style.display = 'block';
    } else if (result === 0) {
        resultSafe.style.display = 'block';
    }

    resultContainer.style.display = 'block';
}

// Fonction générique pour analyser la requête (HTTP, SQL, etc.)
async function analyzeQuery(type) {
    const query = document.getElementById(`${type}-query`).value.trim();
    const selectedModel = document.getElementById(`${type}-model-selector`).value;
    const resultContainer = document.getElementById(`${type}-result-container`);
    const predictBtn = document.getElementById(`${type}-predict-btn`);

    if (!query) {
        alert(`❌ Veuillez entrer une requête ${type.toUpperCase()}`);
        return;
    }

    try {
        // Désactiver le bouton de prédiction et afficher un loader
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Analyse...';

        const response = await fetch(`/predict-${type}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, model: selectedModel })
        });

        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayResult(type, data.result);
        addToHistory(query, data.result, type);

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
// Main
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

document.addEventListener('DOMContentLoaded', () => {
    initSidebar();

    // Initialiser les sélecteurs de modèles pour HTTP et SQL
    initModelSelector('http');
    initModelSelector('sql');

    // Gestion des boutons d'analyse SQL et HTTP
    document.getElementById('sql-predict-btn')?.addEventListener('click', () => analyzeQuery('sql'));
    document.getElementById('http-predict-btn')?.addEventListener('click', () => analyzeQuery('http'));


    // Charts (sur la page /analystic)
    if (window.location.pathname.includes('/analystic')) initCharts();
});

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//SideBar Initialisation
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
    const selectedModel = document.getElementById(`${type}-model-selector`)?.value || 'svm';  // Utiliser le type ici
    let modelLabel = '';
    let modelClass = '';

    switch (selectedModel) {
    case 'randomforest':
        modelLabel = 'Random Forest';
        modelClass = 'rf';
        break;
    case 'svm':
        modelLabel = 'SVM';
        modelClass = 'svm';
        break;
    case 'logisticregression':
        modelLabel = 'Logistic Regression';
        modelClass = 'lr';
        break;
    case 'gradientboosting':
        modelLabel = 'Gradient Boosting';
        modelClass = 'lr';
        break;
    default:
        modelLabel = selectedModel;
        modelClass = '';
}


    // Générer un nouvel ID pour l'historique (ex: #003)
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

    // Affichage du résultat
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

