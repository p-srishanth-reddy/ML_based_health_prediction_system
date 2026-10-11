document.addEventListener('DOMContentLoaded', () => {
    initHome();
    initPrediction();
    initEvaluation();
});

async function initHome() {
    const modulesDiv = document.getElementById('modules');
    if (!modulesDiv) return;

    try {
        const response = await fetch('/api/modules');
        const modules = await response.json();

        modulesDiv.innerHTML = Object.entries(modules).map(([id, info]) => `
            <div class="col-md-6 col-lg-3">
                <div class="card h-100 shadow-sm rounded-4 p-4 text-center">
                    <div class="mb-3">
                        <i class="bi bi-heart-pulse text-primary" style="font-size: 2.5rem;"></i>
                    </div>
                    <h5 class="fw-bold">${info.name}</h5>
                    <p class="small text-muted">${info.description}</p>
                    <a href="/predict/${id}" class="btn btn-outline-primary w-100 mt-auto rounded-pill">Open Screening</a>
                </div>
            </div>
        `).join('');
    } catch (err) {
        console.error('Error loading modules:', err);
    }
}

async function initPrediction() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    const moduleId = window.currentModuleId;
    const nameEl = document.getElementById('module-name');
    const descEl = document.getElementById('module-desc');

    try {
        const response = await fetch(`/api/modules/${moduleId}/schema`);
        const schema = await response.json();

        // Set header
        const modulesResponse = await fetch('/api/modules');
        const modules = await modulesResponse.json();
        const moduleInfo = modules[moduleId];
        nameEl.textContent = moduleInfo.name;
        descEl.textContent = moduleInfo.description;

        // Build form
        form.innerHTML = Object.entries(schema).map(([key, config]) => {
            if (config.type === 'select') {
                return `
                    <div class="col-md-6">
                        <label class="form-label small fw-bold">${config.label}</label>
                        <select name="${key}" class="form-select" required>
                            ${config.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}
                        </select>
                    </div>
                `;
            } else {
                return `
                    <div class="col-md-6">
                        <label class="form-label small fw-bold">${config.label}</label>
                        <input type="number" step="any" name="${key}" class="form-control" required>
                    </div>
                `;
            }
        }).join('');

        document.getElementById('btn-predict').onclick = async () => {
            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = isNaN(value) || value === '' ? value : parseFloat(value);
            });

            const res = await fetch(`/api/predict/${moduleId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await res.json();

            if (result.error) {
                alert('Error: ' + result.error);
                return;
            }

            showResult(result);
        };
    } catch (err) {
        console.error('Error initializing prediction form:', err);
    }
}

async function showResult(res) {
    const panel = document.getElementById('result-panel');
    const content = document.getElementById('prediction-content');

    const rf = res.random_forest;
    const dt = res.decision_tree;

    // Get the module name for a personalized message
    const moduleId = window.currentModuleId;
    let moduleInfo = { name: 'Condition' };
    try {
        const modulesResponse = await fetch('/api/modules').then(r => r.json());
        moduleInfo = modulesResponse[moduleId] || { name: 'Condition' };
    } catch (e) { console.error(e); }

    const conditionName = moduleInfo.name;

    // Use the Random Forest (Primary) for the main message
    const finalScore = rf.score ? rf.score.toFixed(1) : 'N/A';
    const predictionLabel = rf.prediction;

    // Create a user-friendly sentence
    let statusMessage = "";
    if (finalScore !== 'N/A') {
        const scoreNum = parseFloat(finalScore);
        if (scoreNum > 50) {
            statusMessage = `The model suggests you have a <strong class="text-danger">${finalScore}%</strong> risk of ${conditionName}.`;
        } else {
            statusMessage = `The model suggests you have a <strong class="text-success">${finalScore}%</strong> risk of ${conditionName}.`;
        }
    } else {
        statusMessage = `Prediction: ${predictionLabel}`;
    }

    content.innerHTML = `
        <div class="text-center mb-4">
            <div class="h4 fw-bold mb-3">${statusMessage}</div>
            <div class="p-3 border rounded-3 bg-light d-inline-block">
                <div class="small text-muted">AI Model Analysis</div>
                <div class="h6 mb-0">Random Forest Confidence: ${finalScore}%</div>
            </div>
        </div>
    `;
    panel.classList.remove('d-none');
    panel.scrollIntoView({ behavior: 'smooth' });
}

async function initEvaluation() {
    const select = document.getElementById('module-select');
    if (!select) return;

    try {
        const response = await fetch('/api/modules');
        const modules = await response.json();

        select.innerHTML += Object.entries(modules).map(([id, info]) => `
            <option value="${id}">${info.name}</option>
        `).join('');

        select.onchange = async (e) => {
            const mid = e.target.value;
            if (!mid) {
                document.getElementById('evaluation-content').classList.add('d-none');
                document.getElementById('empty-state').classList.remove('d-none');
                return;
            }

            const evRes = await fetch(`/api/evaluation/${mid}`);
            const evData = await evRes.json();

            if (evData.error) {
                alert(evData.error);
                return;
            }

            displayEvaluation(evData);
        };
    } catch (err) {
        console.error('Error initializing evaluation:', err);
    }
}

function displayEvaluation(data) {
    document.getElementById('empty-state').classList.add('d-none');
    document.getElementById('evaluation-content').classList.remove('d-none');

    const info = document.getElementById('dataset-info');
    info.innerHTML = `
        <p class="mb-1"><strong>Model:</strong> ${data.primary_model || 'N/A'}</p>
        <p class="mb-1"><strong>Algorithm:</strong> RF vs DT</p>
    `;

    const body = document.getElementById('metrics-body');
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];

    body.innerHTML = metrics.map(m => `
        <tr>
            <td class="fw-bold text-capitalize">${m.replace('_', ' ')}</td>
            <td>${data.decision_tree[m].toFixed(4)}</td>
            <td>${data.random_forest[m].toFixed(4)}</td>
        </tr>
    `).join('');
}
