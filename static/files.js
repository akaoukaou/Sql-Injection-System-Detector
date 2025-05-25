document.addEventListener('DOMContentLoaded', function() {
    // Stockage des fichiers en mémoire
    let uploadedFiles = [];
    
    // Éléments DOM
    const dropZone = document.querySelector('.drop-zone');
    const fileInput = document.getElementById('fileInput');
    const uploadProgress = document.getElementById('uploadProgress');
    const filesTableBody = document.getElementById('filesTableBody');
    const searchInput = document.getElementById('searchInput');

    // Gestion du Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('highlight');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('highlight');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('highlight');
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // Gestion de la sélection de fichiers
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFiles(fileInput.files);
        }
    });

    // Fonction pour gérer les fichiers
    function handleFiles(files) {
        Array.from(files).forEach(file => {
            if (!file.name.match(/\.(log|csv|json|txt)$/i)) {
                showAlert(`Le type de fichier ${file.name.split('.').pop()} n'est pas supporté`);
                return;
            }
            processFile(file);
        });
    }

    // Traitement du fichier (simulation d'upload)
    function processFile(file) {
        const progressItem = createProgressItem(file);
        
        // Simulation de progression (pour le visuel)
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            updateProgress(progressItem, progress, 100);
            
            if (progress >= 100) {
                clearInterval(interval);
                handleUploadSuccess(progressItem, file);
            }
        }, 100);
    }

    function createProgressItem(file) {
        const progressItem = document.createElement('div');
        progressItem.className = 'progress-item';
        progressItem.innerHTML = `
            <div class="file-info">
                <span>${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
            <div class="progress-bar">
                <div class="progress"></div>
            </div>
            <div class="progress-status">0%</div>
        `;
        uploadProgress.appendChild(progressItem);
        return progressItem;
    }

    function updateProgress(progressItem, loaded, total) {
        const percent = Math.round((loaded / total) * 100);
        progressItem.querySelector('.progress').style.width = `${percent}%`;
        progressItem.querySelector('.progress-status').textContent = `${percent}%`;
    }

    function handleUploadSuccess(progressItem, file) {
        progressItem.classList.add('completed');
        
        // Créer l'objet fichier pour le stockage
        const fileData = {
            id: Date.now().toString(),
            name: file.name,
            type: file.name.split('.').pop().toUpperCase(),
            size: file.size,
            uploadDate: new Date(),
            fileObject: file  // Stocker l'objet File
        };
        
        uploadedFiles.unshift(fileData); // Ajouter au début du tableau
        refreshFileTable();
    }

    function refreshFileTable() {
        filesTableBody.innerHTML = '';
        uploadedFiles.forEach(file => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><i class='bx bx-file'></i>${file.name}</td>
                <td>${file.type}</td>
                <td>${formatFileSize(file.size)}</td>
                <td>${file.uploadDate.toLocaleString()}</td>
                <td>
                    <button class="action-btn analyze-btn" onclick="analyzeFile('${file.id}')">
                        <i class='bx bx-shield-alt'></i>
                    </button>
                    <button class="action-btn download-btn" onclick="downloadFile('${file.id}')">
                        <i class='bx bx-download'></i>
                    </button>
                    <button class="action-btn delete-btn" onclick="deleteFile('${file.id}')">
                        <i class='bx bx-trash'></i>
                    </button>
                </td>
            `;
            filesTableBody.appendChild(row);
        });
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Recherche de fichiers
    searchInput.addEventListener('input', () => {
        const term = searchInput.value.toLowerCase();
        document.querySelectorAll('#filesTableBody tr').forEach(row => {
            const fileName = row.cells[0].textContent.toLowerCase();
            row.style.display = fileName.includes(term) ? '' : 'none';
        });
    });
});

// Fonctions globales
function analyzeFile(fileId) {
    alert('Fonctionnalité d\'analyse bientôt disponible !');
}

function downloadFile(fileId) {
    alert('Fonctionnalité de téléchargement bientôt disponible !');
}

function deleteFile(fileId) {
    if (confirm('Voulez-vous vraiment supprimer ce fichier ?')) {
        alert('Fonctionnalité de suppression bientôt disponible !');
    }
}

function showAlert(message) {
    alert(message);
}