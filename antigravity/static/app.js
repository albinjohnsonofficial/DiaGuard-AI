// Client-Side Logic for DiaGuard Clinical Prediction System

const PRESETS = {
    healthy: {
        Pregnancies: 1,
        Glucose: 88,
        BloodPressure: 68,
        SkinThickness: 20,
        Insulin: 65,
        BMI: 22.4,
        DiabetesPedigreeFunction: 0.21,
        Age: 24
    },
    borderline: {
        Pregnancies: 3,
        Glucose: 132,
        BloodPressure: 78,
        SkinThickness: 28,
        Insulin: 140,
        BMI: 29.8,
        DiabetesPedigreeFunction: 0.45,
        Age: 41
    },
    high_risk: {
        Pregnancies: 6,
        Glucose: 175,
        BloodPressure: 88,
        SkinThickness: 35,
        Insulin: 210,
        BMI: 36.5,
        DiabetesPedigreeFunction: 0.85,
        Age: 52
    }
};

// Synchronize Range Slider to Number Input
function syncInput(featureId, value) {
    const numInput = document.getElementById(featureId);
    if (numInput) {
        numInput.value = value;
    }
}

// Synchronize Number Input to Range Slider
function syncSlider(featureId, value) {
    const slider = document.getElementById(`${featureId}-slider`);
    if (slider) {
        slider.value = value;
    }
}

// Switch navigation tabs
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => 
        b.getAttribute('onclick').includes(tabName)
    );
    if (activeBtn) activeBtn.classList.add('active');

    const targetTab = document.getElementById(`tab-${tabName}`);
    if (targetTab) targetTab.classList.add('active');
}

// Load patient preset
function loadPreset(presetKey) {
    const preset = PRESETS[presetKey];
    if (!preset) return;

    for (const [key, value] of Object.entries(preset)) {
        syncInput(key, value);
        syncSlider(key, value);
    }

    // Automatically trigger prediction for intuitive user experience
    submitPrediction();
}

// Reset form to defaults
function resetForm() {
    const defaults = {
        Pregnancies: 2,
        Glucose: 115,
        BloodPressure: 72,
        SkinThickness: 23,
        Insulin: 80,
        BMI: 28.1,
        DiabetesPedigreeFunction: 0.38,
        Age: 33
    };

    for (const [key, value] of Object.entries(defaults)) {
        syncInput(key, value);
        syncSlider(key, value);
    }

    document.getElementById('empty-state').style.display = 'block';
    document.getElementById('result-content').style.display = 'none';
    const statusBadge = document.getElementById('result-status-badge');
    statusBadge.textContent = 'Awaiting Input';
    statusBadge.className = 'badge';
}

// Submit Prediction Function
async function submitPrediction(e) {
    if (e) e.preventDefault();

    const btn = document.getElementById('btn-predict');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating SVM Margin...';
    btn.disabled = true;

    const payload = {
        Pregnancies: parseFloat(document.getElementById('Pregnancies').value),
        Glucose: parseFloat(document.getElementById('Glucose').value),
        BloodPressure: parseFloat(document.getElementById('BloodPressure').value),
        SkinThickness: parseFloat(document.getElementById('SkinThickness').value),
        Insulin: parseFloat(document.getElementById('Insulin').value),
        BMI: parseFloat(document.getElementById('BMI').value),
        DiabetesPedigreeFunction: parseFloat(document.getElementById('DiabetesPedigreeFunction').value),
        Age: parseFloat(document.getElementById('Age').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!data.success) {
            alert('Prediction error: ' + (data.error || 'Unknown error'));
            return;
        }

        renderResults(data);

    } catch (err) {
        console.error('Error contacting prediction API:', err);
        alert('Failed to connect to backend prediction server. Check console for details.');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// Render Results on the UI
function renderResults(res) {
    document.getElementById('empty-state').style.display = 'none';
    const resultContent = document.getElementById('result-content');
    resultContent.style.display = 'block';

    const statusBadge = document.getElementById('result-status-badge');
    const banner = document.getElementById('diagnosis-banner');
    const badgeCircle = document.getElementById('diag-badge-circle');
    const outcomeText = document.getElementById('diag-outcome-text');
    const subText = document.getElementById('diag-subtext');
    const icon = document.getElementById('diag-icon');

    // Update status badge
    statusBadge.textContent = `${res.risk_category} (${res.probability_diabetic}%)`;
    statusBadge.className = `badge ${res.risk_level === 'high' ? 'badge-danger' : (res.risk_level === 'moderate' ? 'badge-warning' : 'badge-success')}`;

    // Update Banner
    banner.className = `diagnosis-banner diag-${res.risk_level}`;
    badgeCircle.className = `diag-badge-circle diag-${res.risk_level}`;

    if (res.prediction === 1) {
        outcomeText.textContent = 'Diabetic Classification';
        subText.textContent = `Patient biomarkers align with the Diabetic hyperplane boundary (${res.probability_diabetic}% risk).`;
        icon.className = 'fa-solid fa-triangle-exclamation';
    } else {
        outcomeText.textContent = 'Non-Diabetic Classification';
        subText.textContent = `Biomarkers align with non-diabetic profile (${res.probability_non_diabetic}% confidence).`;
        icon.className = 'fa-solid fa-check';
    }

    // Probability Bar
    document.getElementById('prob-diabetic-val').textContent = `${res.probability_diabetic}%`;
    document.getElementById('prob-nondiabetic-val').textContent = `${res.probability_non_diabetic}%`;
    const barFill = document.getElementById('prob-bar-fill');
    barFill.style.width = `${res.probability_diabetic}%`;

    // Contributing Risk Factors
    const factorsList = document.getElementById('risk-factors-list');
    factorsList.innerHTML = '';
    res.risk_factors.forEach(f => {
        const li = document.createElement('li');
        li.className = `sev-${f.severity}`;
        li.textContent = f.text;
        factorsList.appendChild(li);
    });

    // Recommendations
    const recList = document.getElementById('recommendations-list');
    recList.innerHTML = '';
    res.recommendations.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        recList.appendChild(li);
    });
}

// Lightbox for enlarging plots
function openLightbox(src) {
    const lightbox = document.getElementById('lightbox');
    const img = document.getElementById('lightbox-img');
    img.src = src;
    lightbox.classList.add('active');
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    lightbox.classList.remove('active');
}

// Attach event listener
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', submitPrediction);
    }
});
