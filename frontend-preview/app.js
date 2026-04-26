import { createViewer } from './viewer.js';

const API_BASE = 'http://localhost:8000';

const sampleSelect = document.getElementById('sampleSelect');
const fileInput = document.getElementById('fileInput');
const runBtn = document.getElementById('runBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorBox = document.getElementById('errorBox');

const gridImage = document.getElementById('gridImage');
const codeBlock = document.getElementById('codeBlock');
const groundTruthBlock = document.getElementById('groundTruthBlock');
const reasoningBlock = document.getElementById('reasoningBlock');
const execBadge = document.getElementById('execBadge');
const solidBadge = document.getElementById('solidBadge');
const waterBadge = document.getElementById('waterBadge');
const bboxBadge = document.getElementById('bboxBadge');
const confBadge = document.getElementById('confBadge');
const stepLink = document.getElementById('stepLink');
const stlLink = document.getElementById('stlLink');
const copyBtn = document.getElementById('copyBtn');

const viewer = createViewer(document.getElementById('viewer'));

async function loadSamples() {
  const resp = await fetch(`${API_BASE}/samples`);
  const samples = await resp.json();
  sampleSelect.innerHTML = '<option value="">-- choose sample --</option>';
  for (const sample of samples) {
    const opt = document.createElement('option');
    opt.value = sample.sample_id;
    opt.textContent = sample.display_name;
    sampleSelect.appendChild(opt);
  }
}

async function uploadIfNeeded() {
  const file = fileInput.files?.[0];
  if (!file) return null;
  const body = new FormData();
  body.append('file', file);
  const resp = await fetch(`${API_BASE}/upload`, { method: 'POST', body });
  const payload = await resp.json();
  if (!resp.ok || payload.error) throw new Error(payload.error || 'Upload failed');
  return payload.filename;
}

function showError(msg) {
  errorBox.style.display = 'block';
  errorBox.textContent = msg;
}

function clearError() {
  errorBox.style.display = 'none';
  errorBox.textContent = '';
}

runBtn.addEventListener('click', async () => {
  clearError();
  loading.classList.remove('hidden');
  results.style.display = 'none';

  try {
    const filename = await uploadIfNeeded();
    const source = filename ? 'upload' : 'sample';
    const body = {
      source,
      sample_id: source === 'sample' ? sampleSelect.value : null,
      filename: source === 'upload' ? filename : null,
    };
    const resp = await fetch(`${API_BASE}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const payload = await resp.json();
    if (!resp.ok || payload.success === false) {
      throw new Error(`${payload.stage || 'pipeline'}: ${payload.error || 'Unknown error'}`);
    }

    gridImage.src = `${API_BASE}${payload.render_grid_url}`;
    codeBlock.textContent = payload.cadquery_code || '';
    groundTruthBlock.textContent = payload.ground_truth_cadquery_code || '(no ground truth for uploads)';
    reasoningBlock.textContent = payload.llm_reasoning || '';

    execBadge.textContent = `Executes: ${payload.validation?.executes ? 'yes' : 'no'}`;
    solidBadge.textContent = `Solid: ${payload.validation?.produces_solid ? 'yes' : 'no'}`;
    waterBadge.textContent = `Watertight: ${payload.validation?.is_watertight ? 'yes' : 'no'}`;
    bboxBadge.textContent = `BBox(mm): ${(payload.validation?.bounding_box_mm || []).join(' x ')}`;
    confBadge.textContent = `Confidence: ${payload.confidence || 'n/a'}`;

    stepLink.href = `${API_BASE}${payload.step_url}`;
    stlLink.href = `${API_BASE}${payload.stl_url}`;
    viewer.loadStl(`${API_BASE}${payload.stl_url}`);

    results.style.display = 'block';
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    loading.classList.add('hidden');
  }
});

copyBtn.addEventListener('click', async () => {
  await navigator.clipboard.writeText(codeBlock.textContent || '');
});

loadSamples().catch((err) => showError(err.message || String(err)));
