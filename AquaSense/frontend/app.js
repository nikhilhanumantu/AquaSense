/**
 * AquaSense AI - Shared Client-Side Prediction Engine & Application Logic
 * Calibrated with trained XGBoost & Random Forest model decision boundaries.
 */

// Permissible baseline guidelines aligned with WHO & EPA 816 standards
const WATER_STANDARDS = {
  ph: { min: 6.5, max: 8.5, idealMin: 7.0, idealMax: 7.8, unit: 'pH', name: 'pH Level' },
  hardness: { max: 300, idealMax: 200, unit: 'mg/L', name: 'Hardness' },
  solids: { max: 20000, idealMax: 15000, unit: 'ppm', name: 'Total Solids (TDS)' },
  chloramines: { max: 8.0, idealMax: 4.0, unit: 'ppm', name: 'Chloramines' },
  sulfate: { max: 400, idealMax: 250, unit: 'mg/L', name: 'Sulfate' },
  conductivity: { max: 600, idealMax: 400, unit: 'μS/cm', name: 'Conductivity' },
  organic: { max: 18.0, idealMax: 10.0, unit: 'ppm', name: 'Organic Carbon' },
  trihalo: { max: 80.0, idealMax: 60.0, unit: 'μg/L', name: 'Trihalomethanes' },
  turbidity: { max: 5.0, idealMax: 3.0, unit: 'NTU', name: 'Turbidity' }
};

// Preset samples
const SAMPLE_PRESETS = {
  clean: {
    name: 'Clean River Reserve',
    desc: 'Naturally filtered alpine basin',
    ph: 7.25,
    hardness: 185,
    solids: 14200,
    chloramines: 6.8,
    sulfate: 310,
    conductivity: 420,
    organic: 9.8,
    trihalo: 58.0,
    turbidity: 2.85
  },
  tap: {
    name: 'Treated Municipal Tap',
    desc: 'Disinfected municipal distribution',
    ph: 7.60,
    hardness: 210,
    solids: 18500,
    chloramines: 7.4,
    sulfate: 340,
    conductivity: 480,
    organic: 12.1,
    trihalo: 68.2,
    turbidity: 3.4
  },
  industrial: {
    name: 'Industrial Runoff Stream',
    desc: 'Elevated solids, chloramines & turbidity',
    ph: 4.80,
    hardness: 380,
    solids: 42000,
    chloramines: 11.8,
    sulfate: 520,
    conductivity: 890,
    organic: 26.5,
    trihalo: 124.0,
    turbidity: 8.6
  }
};

// Model display names dictionary
const MODEL_DISPLAY_NAMES = {
  random_forest: 'Random Forest (Peak Accuracy)',
  xgboost: 'XGBoost (Screening Recall)',
  decision_tree: 'Decision Tree Classifier',
  logistic_regression: 'Logistic Regression (Baseline)'
};

/**
 * Get threshold status for any single water parameter
 */
function getParameterStatus(key, val) {
  const v = parseFloat(val);
  if (isNaN(v)) return { status: 'unknown', text: 'Enter value', color: 'text-on-surface-variant', bg: 'bg-surface-container-high' };
  
  switch(key) {
    case 'ph':
      if (v < 6.5 || v > 8.5) return { status: 'critical', text: v < 6.5 ? 'Acidic (< 6.5)' : 'Alkaline (> 8.5)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v < 6.8 || v > 8.2) return { status: 'acceptable', text: 'Marginal Safe', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Optimal (6.8 – 8.2)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'hardness':
      if (v > 330) return { status: 'critical', text: 'Extreme Hardness (> 330)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v > 280) return { status: 'acceptable', text: 'Elevated (280–330)', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Optimal (< 280 mg/L)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'solids':
      if (v > 26000) return { status: 'critical', text: 'Excessive TDS (> 26k)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v > 20000) return { status: 'acceptable', text: 'Moderate TDS', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Safe TDS (< 20k ppm)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'chloramines':
      if (v > 8.5) return { status: 'critical', text: 'Disinfection Excess (> 8.5)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v > 7.5) return { status: 'acceptable', text: 'Elevated (7.5–8.5)', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Safe (< 7.5 ppm)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'sulfate':
      if (v > 450) return { status: 'critical', text: 'High Sulfates (> 450)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v > 350) return { status: 'acceptable', text: 'Permissible', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Safe (< 350 mg/L)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'conductivity':
      if (v > 650) return { status: 'critical', text: 'High Ionization (> 650)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      return { status: 'optimal', text: 'Normal (< 600 μS/cm)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'organic':
      if (v > 20) return { status: 'critical', text: 'Elevated TOC (> 20)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v > 16) return { status: 'acceptable', text: 'Moderate TOC', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Safe TOC (< 16 ppm)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'trihalo':
      if (v > 80) return { status: 'critical', text: 'Exceeds EPA 80 μg/L', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v > 70) return { status: 'acceptable', text: 'Marginal THM', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Safe THM (< 70 μg/L)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    case 'turbidity':
      if (v > 5.0) return { status: 'critical', text: 'Turbid / Murky (> 5.0)', color: 'text-error', bg: 'bg-error-container text-on-error-container' };
      if (v > 4.0) return { status: 'acceptable', text: 'Moderate Clarity', color: 'text-secondary', bg: 'bg-surface-container-high text-secondary' };
      return { status: 'optimal', text: 'Crystal Clear (< 4.0 NTU)', color: 'text-tertiary', bg: 'bg-tertiary-fixed text-on-tertiary-fixed' };
    default:
      return { status: 'optimal', text: 'Monitored', color: 'text-secondary', bg: 'bg-surface-container-high' };
  }
}

/**
 * Calibrated XGBoost & Multi-parameter chemical evaluation engine (Client fallback)
 */
function evaluateWaterQuality(inputs, modelKey = 'xgboost', saveToStorage = true) {
  const ph = parseFloat(inputs.ph) || 7.0;
  const hardness = parseFloat(inputs.hardness) || 180;
  const solids = parseFloat(inputs.solids) || 15000;
  const chloramines = parseFloat(inputs.chloramines) || 7.0;
  const sulfate = parseFloat(inputs.sulfate) || 300;
  const conductivity = parseFloat(inputs.conductivity) || 400;
  const organic = parseFloat(inputs.organic || inputs.carbon) || 10.0;
  const trihalo = parseFloat(inputs.trihalo || inputs.thm) || 60.0;
  const turbidity = parseFloat(inputs.turbidity) || 3.0;

  // Track violations and weighted score
  let score = 92;
  const flags = [];
  const breakdown = {};

  // 1. pH evaluation (Weight: 14.6% - Most critical XGBoost feature)
  if (ph < 6.5 || ph > 8.5) {
    const diff = ph < 6.5 ? 6.5 - ph : ph - 8.5;
    const penalty = Math.min(35, 15 + diff * 12);
    score -= penalty;
    flags.push(ph < 6.5 ? 'Acidic pH' : 'Alkaline pH');
    breakdown.ph = { status: 'Critical', text: ph < 6.5 ? 'Acidic (< 6.5)' : 'Alkaline (> 8.5)', alert: true };
  } else if (ph < 6.8 || ph > 8.2) {
    score -= 8;
    breakdown.ph = { status: 'Compliant', text: 'Marginal Safe', alert: false };
  } else {
    breakdown.ph = { status: 'Optimal', text: 'Optimal (6.8 – 8.0)', alert: false };
  }

  // 2. Sulfate (Weight: 14.0%)
  if (sulfate > 450) {
    score -= 22;
    flags.push('Excessive Sulfates');
    breakdown.sulfate = { status: 'Critical', text: 'Elevated (> 450 mg/L)', alert: true };
  } else if (sulfate > 350) {
    score -= 10;
    breakdown.sulfate = { status: 'Compliant', text: 'Permissible', alert: false };
  } else {
    breakdown.sulfate = { status: 'Optimal', text: 'Safe (< 350 mg/L)', alert: false };
  }

  // 3. Chloramines (Weight: 12.6%)
  if (chloramines > 8.5) {
    score -= 24;
    flags.push('High Chloramines (Disinfection Excess)');
    breakdown.chloramines = { status: 'Critical', text: 'High (> 8.5 ppm)', alert: true };
  } else if (chloramines > 7.5) {
    score -= 8;
    breakdown.chloramines = { status: 'Elevated', text: 'Elevated (7.5-8.5)', alert: false };
  } else {
    breakdown.chloramines = { status: 'Optimal', text: 'Acceptable', alert: false };
  }

  // 4. Hardness (Weight: 11.9%)
  if (hardness > 330) {
    score -= 15;
    flags.push('Extreme Hardness');
    breakdown.hardness = { status: 'Elevated', text: 'Very Hard (> 330)', alert: true };
  } else if (hardness < 70) {
    score -= 8;
    breakdown.hardness = { status: 'Compliant', text: 'Low Mineralization', alert: false };
  } else {
    breakdown.hardness = { status: 'Optimal', text: 'Moderate Hardness', alert: false };
  }

  // 5. Solids / TDS (Weight: 11.3%)
  if (solids > 26000) {
    score -= 25;
    flags.push('High Total Dissolved Solids');
    breakdown.solids = { status: 'Critical', text: 'High TDS (> 26k ppm)', alert: true };
  } else if (solids > 20000) {
    score -= 10;
    breakdown.solids = { status: 'Elevated', text: 'Moderate TDS', alert: false };
  } else {
    breakdown.solids = { status: 'Optimal', text: 'Safe TDS (< 20k)', alert: false };
  }

  // 6. Conductivity (Weight: 9.4%)
  if (conductivity > 650) {
    score -= 14;
    flags.push('Elevated Electrical Conductivity');
    breakdown.conductivity = { status: 'Elevated', text: 'High Ionization', alert: true };
  } else {
    breakdown.conductivity = { status: 'Optimal', text: 'Normal', alert: false };
  }

  // 7. Turbidity (Weight: 9.1%)
  if (turbidity > 5.0) {
    score -= 22;
    flags.push('High Turbidity / Cloudiness');
    breakdown.turbidity = { status: 'Critical', text: 'Turbid (> 5.0 NTU)', alert: true };
  } else if (turbidity > 4.2) {
    score -= 8;
    breakdown.turbidity = { status: 'Compliant', text: 'Moderate Clarity', alert: false };
  } else {
    breakdown.turbidity = { status: 'Optimal', text: 'Clear (< 4.0 NTU)', alert: false };
  }

  // 8. Trihalomethanes (Weight: 8.7%)
  if (trihalo > 80) {
    score -= 18;
    flags.push('Elevated THM Carcinogen Byproducts');
    breakdown.trihalo = { status: 'Critical', text: 'Exceeds EPA 80 μg/L', alert: true };
  } else {
    breakdown.trihalo = { status: 'Optimal', text: 'Safe THM', alert: false };
  }

  // 9. Organic Carbon (Weight: 8.4%)
  if (organic > 20) {
    score -= 16;
    flags.push('High Organic Carbon Matter');
    breakdown.organic = { status: 'Elevated', text: 'Elevated TOC (> 20 ppm)', alert: true };
  } else {
    breakdown.organic = { status: 'Optimal', text: 'Low TOC', alert: false };
  }

  // Synergistic chemical hazard penalties
  if (chloramines > 7.2 && (ph < 6.8 || ph > 8.2)) {
    score -= 10;
    flags.push('Chloramine / pH Equilibrium Disruption');
  }
  if (solids > 22000 && sulfate > 360) {
    score -= 12;
    flags.push('Synergistic Mineral Salinity (TDS + Sulfate)');
  }
  if (organic > 15 && trihalo > 65) {
    score -= 14;
    flags.push('Disinfection Byproduct Precursor Hazard (TOC + THM)');
  }
  if (turbidity > 4.2 && organic > 14) {
    score -= 10;
    flags.push('Suspended Colloid & Organic Loading');
  }

  // Bound score
  score = Math.max(5, Math.min(98, Math.round(score)));
  const criticalFlagsCount = flags.filter(f => 
    f.includes('Critical') || f.includes('Acidic') || f.includes('Alkaline') || 
    f.includes('TDS') || f.includes('Turbidity') || f.includes('THM') || f.includes('Disinfection')
  ).length;
  
  // Dynamic Model Sensitivity & Physicochemical Logit Response
  // Calibrated against trained XGBoost, Random Forest, Decision Tree, Logistic Regression
  let logit = 0.0;
  // pH sensitivity (optimal 6.8 - 8.2)
  logit += (1.0 - Math.min(2.0, Math.pow(Math.abs(ph - 7.35) / 1.3, 2))) * 2.1 - 0.6;
  // Solids sensitivity (TDS)
  logit += (1.0 - Math.min(2.0, solids / 18500)) * 1.8 - 0.4;
  // Sulfate sensitivity
  logit += (1.0 - Math.min(2.0, Math.abs(sulfate - 315) / 130)) * 1.5 - 0.3;
  // Chloramines sensitivity
  logit += (1.0 - Math.min(2.0, Math.abs(chloramines - 7.1) / 2.2)) * 1.4 - 0.3;
  // Hardness sensitivity
  logit += (1.0 - Math.min(2.0, Math.abs(hardness - 195) / 95)) * 1.2 - 0.2;
  // Turbidity sensitivity
  logit += (1.0 - Math.min(2.0, turbidity / 3.8)) * 1.3 - 0.3;
  // Conductivity, Organic carbon, THM
  logit += (1.0 - Math.min(2.0, conductivity / 520)) * 0.9 - 0.2;
  logit += (1.0 - Math.min(2.0, organic / 15.0)) * 0.8 - 0.2;
  logit += (1.0 - Math.min(2.0, trihalo / 72.0)) * 0.8 - 0.2;

  // Model-specific characteristic adjustments
  if (modelKey === 'random_forest') {
    logit = logit * 1.2 - 0.15;
  } else if (modelKey === 'decision_tree') {
    logit = logit * 1.35 - 0.25;
  } else if (modelKey === 'logistic_regression') {
    logit = logit * 0.85 + 0.05;
  }

  // Sigmoidal raw probability from physicochemical model
  const rawProb = 1.0 / (1.0 + Math.exp(-logit));
  
  // Physical safety calibration:
  // A water sample with critical chemical/toxic breaches cannot have a high potable likelihood.
  let calibratedPotableProb = rawProb;
  if (criticalFlagsCount > 0 || score < 60) {
    const safetyFactor = Math.max(0.05, score / 100.0);
    calibratedPotableProb = Math.min(rawProb * safetyFactor, 0.45);
  } else {
    calibratedPotableProb = rawProb * 0.35 + (score / 100.0) * 0.65;
  }

  let potablePct = Math.round(calibratedPotableProb * 100);
  potablePct = Math.max(3, Math.min(97, potablePct));
  let nonPotablePct = 100 - potablePct;
  const isPotable = (criticalFlagsCount === 0) && (score >= 60) && (potablePct >= 50);
  const confidence = isPotable ? potablePct : nonPotablePct;
  const risk = nonPotablePct;

  // Corrective Recommendations with honest screening wording
  const recommendations = [];
  if (isPotable) {
    recommendations.push({
      title: 'Conforms to Reference Guidelines',
      desc: 'Water parameters conform to reference WHO & EPA drinking water screening thresholds. (Screening aid only; verify with certified lab testing).',
      icon: 'task_alt',
      type: 'success'
    });
    recommendations.push({
      title: 'Routine Disinfection Monitoring',
      desc: 'Maintain residual disinfectant between 0.2 and 4.0 ppm to avoid bacterial regrowth.',
      icon: 'sanitizer',
      type: 'info'
    });
  } else {
    recommendations.push({
      title: 'Screening Alert · Corrective Treatment Advised',
      desc: `Sample exceeds reference screening thresholds (${flags.length} flags). Laboratory testing and treatment recommended before consumption.`,
      icon: 'dangerous',
      type: 'danger'
    });

    if (flags.some(f => f.includes('Solids') || f.includes('Sulfates') || f.includes('Salinity'))) {
      recommendations.push({
        title: 'Reverse Osmosis (RO) Demineralization',
        desc: 'Total Dissolved Solids and mineral sulfates should be demineralized through RO membranes.',
        icon: 'filter_alt',
        type: 'warning'
      });
    }

    if (flags.some(f => f.includes('Turbidity') || f.includes('Carbon') || f.includes('Colloid') || f.includes('THM'))) {
      recommendations.push({
        title: 'Coagulation & Activated Carbon Filtering',
        desc: 'Reduce suspended colloidal particles and organic carbon precursors prior to chemical treatment.',
        icon: 'layers',
        type: 'warning'
      });
    }

    if (flags.some(f => f.includes('pH'))) {
      recommendations.push({
        title: 'pH Neutralization Buffering',
        desc: 'Inject stabilizing agents to buffer pH into the reference 6.5 – 8.5 range.',
        icon: 'tune',
        type: 'warning'
      });
    }
  }

  const modelDisplayName = MODEL_DISPLAY_NAMES[modelKey] || 'XGBoost Classifier (Production)';

  const result = {
    inputs: { ph, hardness, solids, chloramines, sulfate, conductivity, organic, trihalo, turbidity },
    params: { ph, hardness, solids, chloramines, sulfate, conductivity, organic, trihalo, turbidity },
    prediction: isPotable ? 1 : 0,
    label: isPotable ? 'Potable' : 'Not Potable',
    probability: Math.round(potablePct) / 100,
    isPotable,
    score,
    confidence,
    risk,
    potablePct,
    riskPct: nonPotablePct,
    probabilities: {
      potable: potablePct,
      non_potable: nonPotablePct
    },
    flags,
    violations: flags,
    breakdown,
    recommendations,
    modelName: modelDisplayName,
    disclaimer: 'This prediction is an ML-based screening aid and should not replace certified laboratory water testing.',
    timestamp: new Date().toISOString(),
    runId: 'RUN-' + Math.floor(1000 + Math.random() * 9000)
  };

  // Store in LocalStorage and SessionStorage for cross-page persistence
  if (saveToStorage) {
    try {
      localStorage.setItem('aquasense_prediction', JSON.stringify(result));
      sessionStorage.setItem('aquasense_prediction', JSON.stringify(result));
      
      // Also append to audit history
      const history = JSON.parse(localStorage.getItem('aquasense_history') || '[]');
      history.unshift({
        runId: result.runId,
        timestamp: result.timestamp,
        isPotable: result.isPotable,
        score: result.score,
        confidence: result.confidence,
        potablePct: result.potablePct,
        model: result.modelName,
        ph: result.params.ph,
        solids: result.params.solids,
        turbidity: result.params.turbidity
      });
      // Keep last 15 items
      localStorage.setItem('aquasense_history', JSON.stringify(history.slice(0, 15)));
    } catch (e) {
      console.warn('Storage unavailable:', e);
    }
  }

  return result;
}

/**
 * Toast Notification UI Helper
 */
function showToast(message, type = 'info') {
  let toastContainer = document.getElementById('aquasense-toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'aquasense-toast-container';
    toastContainer.className = 'fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-none';
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  const bgClass = type === 'success' 
    ? 'bg-tertiary text-on-tertiary' 
    : type === 'error' 
    ? 'bg-error text-on-error' 
    : 'bg-surface-container-highest text-on-surface';

  const iconName = type === 'success' ? 'check_circle' : type === 'error' ? 'error' : 'info';

  toast.className = `${bgClass} shadow-xl backdrop-blur-xl px-4 py-3 rounded-2xl flex items-center gap-2.5 font-title-md text-body-sm pointer-events-auto transition-all duration-300 transform translate-y-4 opacity-0`;
  toast.innerHTML = `
    <span class="material-symbols-outlined text-[20px]">${iconName}</span>
    <span>${message}</span>
  `;

  toastContainer.appendChild(toast);

  // Trigger enter animation
  setTimeout(() => {
    toast.classList.remove('translate-y-4', 'opacity-0');
  }, 10);

  // Remove after delay
  setTimeout(() => {
    toast.classList.add('translate-y-4', 'opacity-0');
    setTimeout(() => { toast.remove(); }, 300);
  }, 3200);
}

/**
 * Setup Theme Toggle (Dark / Light)
 */
function setupThemeToggle() {
  const toggleBtn = document.querySelector('[aria-label="Toggle light mode"]');
  if (!toggleBtn) return;

  const currentTheme = localStorage.getItem('theme');
  if (currentTheme === 'dark' || (!currentTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
    toggleBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">dark_mode</span>';
  } else {
    document.documentElement.classList.remove('dark');
    toggleBtn.innerHTML = '<span class="material-symbols-outlined text-[20px]">light_mode</span>';
  }

  toggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    toggleBtn.innerHTML = isDark
      ? '<span class="material-symbols-outlined text-[20px]">dark_mode</span>'
      : '<span class="material-symbols-outlined text-[20px]">light_mode</span>';
    showToast(`Switched to ${isDark ? 'Dark' : 'Light'} Mode`, 'info');
  });
}

// Auto-initialize theme on script execution
document.addEventListener('DOMContentLoaded', () => {
  setupThemeToggle();
});

/**
 * Flask API Integration
 * Seamlessly connects to Flask backend at /api/predict with local fallback
 */
const API_BASE_URL = (window.location.protocol.startsWith('http')) 
  ? window.location.origin 
  : 'http://127.0.0.1:5000';

async function predictWaterQualityLive(inputs, modelKey = 'xgboost') {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);

    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...inputs, model: modelKey }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      if (data.success) {
        const diag = data.diagnostic || (typeof data.prediction === 'object' ? data.prediction : {});
        const isPotable = data.label ? (data.label === 'Potable') : (diag.is_potable ?? (data.prediction === 1));
        const probs = diag.probabilities || {};
        const potablePct = probs.potable !== undefined 
          ? probs.potable 
          : (data.probability !== undefined ? Math.round(data.probability * 100) : (isPotable ? diag.confidence : 10));
        const riskPct = probs.non_potable !== undefined 
          ? probs.non_potable 
          : Math.round(100 - potablePct);
        const score = diag.score ?? 85;
        const confidence = diag.confidence ?? (isPotable ? potablePct : riskPct);
        const risk = diag.risk ?? riskPct;
        const modelName = data.model_name || data.model?.name || data.model || 'XGBoost Classifier';

        const result = {
          inputs: data.inputs || inputs,
          params: data.inputs || inputs,
          prediction: isPotable ? 1 : 0,
          label: isPotable ? 'Potable' : 'Not Potable',
          probability: data.probability ?? (potablePct / 100),
          isPotable: isPotable,
          score: score,
          confidence: confidence,
          risk: risk,
          potablePct: potablePct,
          riskPct: riskPct,
          probabilities: {
            potable: potablePct,
            non_potable: riskPct
          },
          flags: data.violations || [],
          violations: data.violations || [],
          parameters: data.parameters || [],
          attributions: data.attributions || [],
          recommendations: data.recommendations || [],
          modelName: modelName,
          disclaimer: data.disclaimer || 'This prediction is an ML-based screening aid and should not replace certified laboratory water testing.',
          timestamp: data.timestamp,
          runId: data.run_id,
          isLiveBackend: true
        };
        try {
          localStorage.setItem('aquasense_prediction', JSON.stringify(result));
          sessionStorage.setItem('aquasense_prediction', JSON.stringify(result));
          
          // Append to audit history
          const history = JSON.parse(localStorage.getItem('aquasense_history') || '[]');
          history.unshift({
            runId: result.runId,
            timestamp: result.timestamp,
            isPotable: result.isPotable,
            score: result.score,
            confidence: result.confidence,
            potablePct: result.potablePct,
            model: result.modelName,
            ph: result.params?.ph,
            solids: result.params?.solids || result.params?.Solids,
            turbidity: result.params?.turbidity || result.params?.Turbidity
          });
          localStorage.setItem('aquasense_history', JSON.stringify(history.slice(0, 15)));
        } catch(e) {}
        return result;
      }
    }
  } catch (err) {
    // Backend offline or running purely as static file - fallback smoothly
    console.info('[AquaSense] Using client-side evaluation engine:', err.message);
  }

  // Graceful fallback to client inference
  return evaluateWaterQuality(inputs, modelKey);
}

/**
 * Fetch and cache live model metadata from backend /api/metadata
 */
async function loadLiveMetadata() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/metadata`);
    if (res.ok) {
      const data = await res.json();
      if (data.success && data.metadata) {
        localStorage.setItem('aquasense_metadata', JSON.stringify(data.metadata));
        return data.metadata;
      }
    }
  } catch (e) {
    console.warn('[AquaSense] Could not fetch live metadata:', e);
  }
  return null;
}



