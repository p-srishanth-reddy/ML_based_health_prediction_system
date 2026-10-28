document.addEventListener('DOMContentLoaded', () => {
    initHome();
    initPrediction();
    initEvaluation();
});

// Icon and color mapping per module
const MODULE_THEMES = {
    'blood_pressure': { icon: 'bi-heart-pulse-fill', color: 'text-danger', border: 'border-danger-subtle' },
    'diabetes': { icon: 'bi-droplet-half', color: 'text-primary', border: 'border-primary-subtle' },
    'anemia': { icon: 'bi-activity', color: 'text-danger', border: 'border-danger-subtle' },
    'obesity': { icon: 'bi-person-bounding-box', color: 'text-success', border: 'border-success-subtle' }
};

/* =========================================================================
   1. Home Page Initialization
   ========================================================================= */
async function initHome() {
    const modulesDiv = document.getElementById('modules');
    if (!modulesDiv) return;

    try {
        const response = await fetch('/api/modules');
        const modules = await response.json();

        modulesDiv.innerHTML = Object.entries(modules).map(([id, info]) => {
            const theme = MODULE_THEMES[id] || { icon: 'bi-heart-pulse', color: 'text-primary', border: '' };
            return `
                <div class="col-md-6 col-lg-3">
                    <div class="card module-card h-100 rounded-4 p-4 text-center border shadow-sm d-flex flex-column">
                        <div class="mb-3">
                            <span class="d-inline-flex align-items-center justify-content-center p-3 rounded-circle bg-light shadow-sm">
                                <i class="bi ${theme.icon} ${theme.color}" style="font-size: 2rem;"></i>
                            </span>
                        </div>
                        <span class="badge bg-light text-secondary border align-self-center mb-2 px-2 py-1 small">
                            ${info.category || 'Screening'}
                        </span>
                        <h5 class="fw-bold text-navy mb-2">${info.name}</h5>
                        <p class="small text-muted mb-4 flex-grow-1">${info.description}</p>
                        <a href="/predict/${id}" class="btn btn-primary w-100 rounded-pill py-2 shadow-sm">
                            <i class="bi bi-play-circle me-1"></i>Open Screening
                        </a>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loading modules:', err);
    }
}

/* =========================================================================
   2. Prediction Page & Explainability
   ========================================================================= */
let currentModuleData = null;

async function initPrediction() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    const moduleId = window.currentModuleId;
    const nameEl = document.getElementById('module-name');
    const descEl = document.getElementById('module-desc');
    const catEl = document.getElementById('module-category');

    try {
        const modulesResponse = await fetch('/api/modules');
        const modules = await modulesResponse.json();
        currentModuleData = modules[moduleId];

        if (!currentModuleData) {
            form.innerHTML = '<div class="alert alert-danger">Module not found.</div>';
            return;
        }

        nameEl.textContent = currentModuleData.name;
        descEl.textContent = currentModuleData.description;
        if (catEl && currentModuleData.category) catEl.textContent = currentModuleData.category;

        const schema = currentModuleData.schema;

        // Render form input elements
        form.innerHTML = Object.entries(schema).map(([key, config]) => {
            const unitBadge = config.unit ? `<span class="badge bg-light text-secondary border ms-1">${config.unit}</span>` : '';
            const tooltip = config.tooltip ? `<small class="text-muted d-block mt-1">${config.tooltip}</small>` : '';

            if (config.type === 'select') {
                const optionsHtml = config.options.map(opt => {
                    const label = (config.option_labels && config.option_labels[opt]) ? config.option_labels[opt] : opt;
                    const isSelected = (config.default === opt) ? 'selected' : '';
                    return `<option value="${opt}" ${isSelected}>${label}</option>`;
                }).join('');

                return `
                    <div class="col-md-6">
                        <label class="form-label small fw-bold text-navy mb-1">
                            ${config.label} ${unitBadge}
                        </label>
                        <select name="${key}" class="form-select rounded-3 py-2" required>
                            ${optionsHtml}
                        </select>
                        ${tooltip}
                    </div>
                `;
            } else {
                return `
                    <div class="col-md-6">
                        <label class="form-label small fw-bold text-navy mb-1">
                            ${config.label} ${unitBadge}
                        </label>
                        <input type="number"
                               step="${config.step || 'any'}"
                               min="${config.min !== undefined ? config.min : ''}"
                               max="${config.max !== undefined ? config.max : ''}"
                               value="${config.default !== undefined ? config.default : ''}"
                               name="${key}"
                               class="form-control rounded-3 py-2"
                               placeholder="e.g., ${config.default || ''}"
                               required>
                        ${tooltip}
                    </div>
                `;
            }
        }).join('');

        // Wire Sample Value Loaders
        const btnNormal = document.getElementById('btn-sample-normal');
        const btnElevated = document.getElementById('btn-sample-elevated');

        if (btnNormal && currentModuleData.sample_cases && currentModuleData.sample_cases.normal) {
            btnNormal.onclick = () => populateForm(currentModuleData.sample_cases.normal.data);
        }
        if (btnElevated && currentModuleData.sample_cases && currentModuleData.sample_cases.elevated) {
            btnElevated.onclick = () => populateForm(currentModuleData.sample_cases.elevated.data);
        }

        // Handle Predict Button
        const btnPredict = document.getElementById('btn-predict');
        btnPredict.onclick = async () => {
            hideValidationAlert();
            const formData = new FormData(form);
            const payload = {};

            formData.forEach((value, key) => {
                const schemaConfig = schema[key];
                if (schemaConfig && schemaConfig.type === 'number') {
                    payload[key] = (value === '' || isNaN(value)) ? value : parseFloat(value);
                } else {
                    payload[key] = value;
                }
            });

            // Set loading state
            btnPredict.disabled = true;
            btnPredict.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing Patient Data...`;

            try {
                const res = await fetch(`/api/predict/${moduleId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await res.json();

                if (!res.ok || result.error) {
                    showValidationAlert(result.error, result.details || []);
                    return;
                }

                showResult(result);
                saveLocalHistory(moduleId, currentModuleData.name, result);
            } catch (err) {
                console.error(err);
                showValidationAlert('Network error occurred during model inference.');
            } finally {
                btnPredict.disabled = false;
                btnPredict.innerHTML = `<i class="bi bi-cpu me-2"></i>Run Multi-Model Risk Screening`;
            }
        };

    } catch (err) {
        console.error('Error initializing prediction form:', err);
    }
}

function populateForm(data) {
    const form = document.getElementById('prediction-form');
    if (!form || !data) return;
    Object.entries(data).forEach(([key, val]) => {
        const input = form.elements[key];
        if (input) input.value = val;
    });
    hideValidationAlert();
}

function showValidationAlert(errorMsg, details = []) {
    const alertBox = document.getElementById('validation-alert');
    const list = document.getElementById('validation-errors-list');
    if (!alertBox || !list) return;

    list.innerHTML = details.length > 0
        ? details.map(d => `<li>${d}</li>`).join('')
        : `<li>${errorMsg}</li>`;
    alertBox.classList.remove('d-none');
    alertBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideValidationAlert() {
    const alertBox = document.getElementById('validation-alert');
    if (alertBox) alertBox.classList.add('d-none');
}

/* =========================================================================
   Show Result & Explainability Rendering
   ========================================================================= */
function showResult(res) {
    const panel = document.getElementById('result-panel');
    const content = document.getElementById('prediction-content');
    const badgeContainer = document.getElementById('result-badge-container');

    const primaryRes = res.primary_result || {};
    const riskTier = res.risk_tier || primaryRes.human_label || 'Screening Complete';
    const riskBadge = res.risk_badge || 'primary';
    const scoreVal = primaryRes.score !== undefined ? `${primaryRes.score}%` : 'N/A';
    const conditionName = (currentModuleData && currentModuleData.name) || 'Condition';

    // Risk category badge
    badgeContainer.innerHTML = `
        <span class="badge bg-${riskBadge} px-3 py-2 rounded-pill fs-6 fw-semibold">
            ${riskTier}
        </span>
    `;

    // Multi-model comparisons
    const comparisons = res.model_comparisons || {};
    const modelComparisonCards = Object.entries(comparisons).map(([mName, mRes]) => {
        const isPrimary = (mName === res.primary_model);
        const nameClean = mName.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
        const borderStyle = isPrimary ? 'border-primary border-2 bg-light' : 'border';
        return `
            <div class="col-md-4">
                <div class="card h-100 rounded-3 p-3 text-center ${borderStyle}">
                    <div class="d-flex align-items-center justify-content-center gap-1 mb-1">
                        <span class="small fw-bold text-navy">${nameClean}</span>
                        ${isPrimary ? '<span class="badge bg-primary rounded-pill small" style="font-size:0.65rem;">Primary</span>' : ''}
                    </div>
                    <div class="h5 fw-bold mb-1 text-navy">${mRes.score}%</div>
                    <div class="small text-muted">${mRes.human_label}</div>
                </div>
            </div>
        `;
    }).join('');

    // Patient Feature Influences
    const factors = res.contributing_factors || [];
    const factorRows = factors.map(f => {
        const isIncrease = (f.direction === 'increases_risk');
        const badgeColor = isIncrease ? 'text-danger bg-danger-subtle' : 'text-success bg-success-subtle';
        const icon = isIncrease ? 'bi-arrow-up-right' : 'bi-arrow-down-right';
        const barColor = isIncrease ? 'factor-increases' : 'factor-reduces';

        return `
            <div class="mb-3 p-3 bg-light rounded-3 border">
                <div class="d-flex align-items-center justify-content-between mb-1">
                    <div>
                        <strong class="text-navy">${f.name}</strong>
                        <span class="badge bg-white text-secondary border ms-2">${f.value}</span>
                        <small class="text-muted ms-2">${f.reference}</small>
                    </div>
                    <span class="badge ${badgeColor} px-2 py-1 small d-flex align-items-center gap-1">
                        <i class="bi ${icon}"></i> ${isIncrease ? 'Elevates Risk' : 'Reduces Risk'}
                    </span>
                </div>
                <div class="progress factor-bar mb-1" style="height: 6px;">
                    <div class="progress-bar ${barColor}" role="progressbar" style="width: ${Math.min(100, f.impact_score * 2.5)}%"></div>
                </div>
                <small class="text-muted">${f.description}</small>
            </div>
        `;
    }).join('');

    content.innerHTML = `
        <!-- Main Score Hero -->
        <div class="text-center mb-5 pb-3 border-bottom">
            <span class="badge bg-light text-secondary border px-3 py-1 rounded-pill small mb-2">
                Uncalibrated Machine Learning Score
            </span>
            <div class="display-3 fw-bold text-navy mb-1">${scoreVal}</div>
            <h4 class="fw-bold mb-2 ${riskBadge === 'danger' ? 'text-danger' : (riskBadge === 'warning' ? 'text-warning' : 'text-success')}">
                ${riskTier}
            </h4>
            <p class="text-muted small mx-auto mb-0" style="max-width: 650px;">
                ${res.recommendation}
            </p>
        </div>

        <!-- Algorithm Consensus Comparison -->
        <div class="mb-5">
            <h6 class="fw-bold text-navy mb-3"><i class="bi bi-diagram-3 text-primary me-2"></i>Multi-Algorithm Output Comparison</h6>
            <div class="row g-3">
                ${modelComparisonCards}
            </div>
        </div>

        <!-- Patient Feature Influences -->
        ${factors.length > 0 ? `
            <div class="mb-4">
                <div class="d-flex align-items-center justify-content-between mb-3">
                    <h6 class="fw-bold text-navy mb-0">
                        <i class="bi bi-sliders text-primary me-2"></i>Key Contributing Indicators
                    </h6>
                    <small class="text-muted">Relative patient deviation from healthy reference baseline</small>
                </div>
                ${factorRows}
                <div class="p-2 bg-light rounded text-muted small mt-2">
                    <i class="bi bi-lightbulb me-1"></i><strong>Statistical Association:</strong>
                    Features marked with elevated risk represent strong statistical correlations within training cohorts, not verified physiological causation.
                </div>
            </div>
        ` : ''}
    `;

    panel.classList.remove('d-none');
    panel.scrollIntoView({ behavior: 'smooth' });
}

/* =========================================================================
   3. Evaluation Page Logic
   ========================================================================= */
async function initEvaluation() {
    const select = document.getElementById('module-select');
    if (!select) return;

    try {
        const response = await fetch('/api/modules');
        const modules = await response.json();

        select.innerHTML = '<option value="">Select a Health Condition to Evaluate...</option>' +
            Object.entries(modules).map(([id, info]) => `
                <option value="${id}">${info.name} (${info.category || 'Clinical'})</option>
            `).join('');

        select.onchange = async (e) => {
            const mid = e.target.value;
            if (!mid) {
                document.getElementById('evaluation-content').classList.add('d-none');
                document.getElementById('empty-state').classList.remove('d-none');
                return;
            }

            try {
                const evRes = await fetch(`/api/evaluation/${mid}`);
                const evData = await evRes.json();

                if (evData.error) {
                    alert(evData.error);
                    return;
                }

                displayEvaluation(evData);
            } catch (err) {
                console.error(err);
                alert('Failed to load evaluation report.');
            }
        };
    } catch (err) {
        console.error('Error initializing evaluation:', err);
    }
}

function displayEvaluation(data) {
    document.getElementById('empty-state').classList.add('d-none');
    document.getElementById('evaluation-content').classList.remove('d-none');

    // Primary model badge
    const badge = document.getElementById('best-model-badge');
    const primaryName = (data.primary_model || 'random_forest').replace('_', ' ').toUpperCase();
    badge.textContent = `Recommended Primary: ${primaryName}`;

    // Dataset Health Information
    const info = document.getElementById('dataset-info');
    const ds = data.dataset_info || {};
    info.innerHTML = `
        <div class="mb-2 d-flex justify-content-between small">
            <span class="text-muted">Total Dataset Cohort:</span>
            <strong class="text-navy">${ds.total_samples || 'N/A'} records</strong>
        </div>
        <div class="mb-2 d-flex justify-content-between small">
            <span class="text-muted">Training Split (80%):</span>
            <span class="fw-semibold">${ds.train_samples || 'N/A'} samples</span>
        </div>
        <div class="mb-2 d-flex justify-content-between small">
            <span class="text-muted">Held-out Test Split (20%):</span>
            <span class="fw-semibold">${ds.test_samples || 'N/A'} samples</span>
        </div>
        <div class="mb-2 d-flex justify-content-between small">
            <span class="text-muted">Duplicate Safeguard:</span>
            <span class="badge bg-success-subtle text-success">Deduplicated</span>
        </div>
        <div class="mt-3 pt-2 border-top small text-muted">
            Stratified partitioning guarantees balanced representation with zero cross-split record leakage.
        </div>
    `;

    // Comparative Performance Metrics Table
    const body = document.getElementById('metrics-body');
    const evalDict = data.evaluation || {};
    const lr = evalDict.logistic_regression || {};
    const dt = evalDict.decision_tree || {};
    const rf = evalDict.random_forest || {};

    const metrics = [
        { key: 'accuracy', label: 'Overall Accuracy' },
        { key: 'balanced_accuracy', label: 'Balanced Accuracy' },
        { key: 'precision', label: 'Precision (Positive Predictive Value)' },
        { key: 'recall', label: 'Recall / Sensitivity' },
        { key: 'specificity', label: 'Specificity (True Negative Rate)' },
        { key: 'f1_score', label: 'F1-Score (Harmonic Mean)' },
        { key: 'roc_auc', label: 'ROC-AUC (Discrimination)' },
        { key: 'cv_f1_mean', label: '5-Fold Cross-Validation F1' }
    ];

    body.innerHTML = metrics.map(m => {
        const lrVal = lr[m.key] !== undefined && lr[m.key] !== null ? lr[m.key].toFixed(4) : 'N/A';
        const dtVal = dt[m.key] !== undefined && dt[m.key] !== null ? dt[m.key].toFixed(4) : 'N/A';
        const rfVal = rf[m.key] !== undefined && rf[m.key] !== null ? rf[m.key].toFixed(4) : 'N/A';

        return `
            <tr>
                <td class="fw-semibold text-navy">${m.label}</td>
                <td><span class="badge bg-light text-dark border font-monospace">${lrVal}</span></td>
                <td><span class="badge bg-light text-dark border font-monospace">${dtVal}</span></td>
                <td><span class="badge bg-light text-dark border font-monospace">${rfVal}</span></td>
            </tr>
        `;
    }).join('');

    // Confusion Matrix Visual Grid
    const cmContainer = document.getElementById('confusion-matrix-container');
    const cmTitle = document.getElementById('cm-model-title');
    const selectedModel = evalDict[data.primary_model] || rf;
    const cm = selectedModel.confusion_matrix || [];

    cmTitle.textContent = `${primaryName} on Test Partition (${ds.test_samples || 0} samples)`;

    if (cm.length === 2) {
        const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
        const total = tn + fp + fn + tp;
        cmContainer.innerHTML = `
            <div class="table-responsive text-center">
                <table class="cm-table">
                    <tr>
                        <td></td>
                        <td class="small fw-bold text-muted">Pred: Negative (0)</td>
                        <td class="small fw-bold text-muted">Pred: Positive (1)</td>
                    </tr>
                    <tr>
                        <td class="small fw-bold text-muted pe-2">Actual: Negative (0)</td>
                        <td>
                            <div class="cm-cell cm-tn">
                                <span class="fs-5">${tn}</span>
                                <small style="font-size: 0.7rem;">True Negative (${((tn/total)*100).toFixed(1)}%)</small>
                            </div>
                        </td>
                        <td>
                            <div class="cm-cell cm-fp">
                                <span class="fs-5">${fp}</span>
                                <small style="font-size: 0.7rem;">False Positive (${((fp/total)*100).toFixed(1)}%)</small>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td class="small fw-bold text-muted pe-2">Actual: Positive (1)</td>
                        <td>
                            <div class="cm-cell cm-fn">
                                <span class="fs-5">${fn}</span>
                                <small style="font-size: 0.7rem;">False Negative (${((fn/total)*100).toFixed(1)}%)</small>
                            </div>
                        </td>
                        <td>
                            <div class="cm-cell cm-tp">
                                <span class="fs-5">${tp}</span>
                                <small style="font-size: 0.7rem;">True Positive (${((tp/total)*100).toFixed(1)}%)</small>
                            </div>
                        </td>
                    </tr>
                </table>
            </div>
        `;
    } else if (cm.length > 2) {
        // Multi-class (Obesity)
        const classes = data.classes || [];
        cmContainer.innerHTML = `
            <div class="table-responsive small">
                <table class="table table-sm table-bordered text-center align-middle mb-0">
                    <thead class="table-light">
                        <tr>
                            <th class="text-start">Actual \\ Predicted</th>
                            ${classes.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${cm.map((row, i) => `
                            <tr>
                                <th class="text-start table-light">${(classes[i] || i).replace(/_/g, ' ')}</th>
                                ${row.map((val, j) => `
                                    <td class="${i === j ? 'bg-success-subtle fw-bold text-success' : (val > 0 ? 'bg-danger-subtle text-danger' : '')}">
                                        ${val}
                                    </td>
                                `).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    // Top Feature Importance Progress Bars
    const fiContainer = document.getElementById('features-importance-container');
    const fiDict = data.feature_importance || {};
    const topFeatures = fiDict[data.primary_model] || fiDict.random_forest || fiDict.decision_tree || [];

    if (topFeatures.length > 0) {
        fiContainer.innerHTML = topFeatures.map(([name, weight]) => {
            const cleanName = name.replace(/_/g, ' ');
            const pct = Math.round(weight * 100);
            return `
                <div class="mb-3">
                    <div class="d-flex justify-content-between small fw-semibold text-navy mb-1">
                        <span>${cleanName}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="progress" style="height: 7px;">
                        <div class="progress-bar bg-primary" role="progressbar" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        }).join('');
    } else {
        fiContainer.innerHTML = '<p class="text-muted small">Feature weights not available.</p>';
    }
}

/* =========================================================================
   4. Local Screening History Management (Privacy-First)
   ========================================================================= */
function saveLocalHistory(moduleId, moduleName, result) {
    try {
        let history = JSON.parse(localStorage.getItem('healthai_screenings') || '[]');
        const entry = {
            id: Date.now(),
            timestamp: new Date().toLocaleString(),
            module_id: moduleId,
            module_name: moduleName,
            prediction: result.primary_result ? result.primary_result.human_label : 'Completed',
            score: result.primary_result ? `${result.primary_result.score}%` : 'N/A',
            risk_tier: result.risk_tier || 'Screened',
            badge: result.risk_badge || 'primary'
        };
        history.unshift(entry);
        if (history.length > 25) history.pop();
        localStorage.setItem('healthai_screenings', JSON.stringify(history));
    } catch (e) {
        console.error('Failed to save screening to localStorage', e);
    }
}

async function loadHistory() {
    const container = document.getElementById('history-container');
    if (!container) return;

    let localHistory = [];
    try {
        localHistory = JSON.parse(localStorage.getItem('healthai_screenings') || '[]');
    } catch (e) {
        console.error(e);
    }

    if (localHistory.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4 text-muted">
                <i class="bi bi-folder2-open display-4 opacity-50"></i>
                <p class="mt-2 mb-0">No screening sessions recorded yet.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="table-responsive">
            <table class="table table-hover align-middle small mb-0">
                <thead class="table-light">
                    <tr>
                        <th>Date & Time</th>
                        <th>Condition Module</th>
                        <th>Risk Assessment</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    ${localHistory.map(h => `
                        <tr>
                            <td class="text-muted">${h.timestamp}</td>
                            <td class="fw-semibold text-navy">${h.module_name}</td>
                            <td><span class="badge bg-${h.badge}-subtle text-${h.badge} border">${h.risk_tier}</span></td>
                            <td class="fw-bold">${h.score}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

async function clearHistory() {
    try {
        localStorage.removeItem('healthai_screenings');
        await fetch('/api/history', { method: 'DELETE' });
    } catch (e) {
        console.error(e);
    }
    loadHistory();
}
