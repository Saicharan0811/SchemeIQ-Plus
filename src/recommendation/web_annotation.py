# -*- coding: utf-8 -*-
"""
src/recommendation/web_annotation.py — Minimal Local Web Annotation Interface (Phase 7F.1)

Provides a lightweight, browser-based user interface for collecting REAL expert
relevance judgments for SchemeIQ+ Learning-to-Rank:
1. Loads citizen archetype profiles from real_archetypes.json.
2. Reuses existing SchemeEligibilityEvaluator and ExpertAnnotationSession.
3. Pre-filters candidate schemes strictly to ELIGIBLE and POTENTIALLY_ELIGIBLE.
4. Displays one archetype at a time with full demographics and narrative.
5. Provides explicit buttons for relevance grades:
     1 = Broadly Relevant
     2 = High Interest
     3 = Primary Choice
   with optional textual justifications.
6. Displays visual progress (e.g. "Archetype 3 of 16").
7. Prevents submission without explicit relevance grades (zero auto-inferred labels).
8. Prevents duplicate judgments while allowing reviewers to update their ratings.
9. Compiles annotations into the Phase 7B RankingDataset schema and validates
   via RankingDatasetValidator before saving to disk.
10. Preserves strict non-PII annotator identifier validation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request

from src.eligibility.evaluator import SchemeEligibilityEvaluator
from src.eligibility.schemas import EligibilityStatus
from src.recommendation.annotation_tool import (
    ELIGIBLE_CANDIDATE_STATUSES,
    VALID_EXPERT_RELEVANCE_GRADES,
    ExpertAnnotationSession,
    load_archetypes_from_file,
    validate_annotator_id,
)
from src.recommendation.dataset_io import RankingDatasetIO
from src.recommendation.dataset_validator import RankingDatasetValidator
from src.recommendation.schemas import CitizenArchetype

logger = logging.getLogger(__name__)

# Minimal self-contained responsive HTML/CSS/JS template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SchemeIQ+ Expert Relevance Annotation</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --accent: #0284c7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 4rem;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--border);
      padding: 1rem 2rem;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .header-content {
      max-width: 1100px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .logo-area h1 { font-size: 1.25rem; font-weight: 700; color: var(--primary); }
    .logo-area p { font-size: 0.85rem; color: var(--text-muted); }
    .header-meta {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-primary { background: #dbeafe; color: #1e40af; }
    .badge-success { background: #dcfce7; color: #166534; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-info { background: #e0f2fe; color: #0369a1; }
    .badge-danger { background: #fee2e2; color: #991b1b; }
    
    .progress-bar-container {
      width: 100%;
      background: #e2e8f0;
      height: 6px;
      position: absolute;
      bottom: 0;
      left: 0;
    }
    .progress-bar {
      height: 100%;
      background: var(--primary);
      transition: width 0.3s ease;
    }

    .container {
      max-width: 1100px;
      margin: 2rem auto;
      padding: 0 1.5rem;
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    .archetype-title-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.75rem;
      gap: 1rem;
    }
    .archetype-title-row h2 {
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--text);
    }
    .archetype-id-tag {
      font-size: 0.9rem;
      color: var(--text-muted);
      font-weight: 500;
      font-family: monospace;
      background: #f1f5f9;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
    }
    .narrative-box {
      background: #f8fafc;
      border-left: 4px solid var(--primary);
      padding: 1rem;
      border-radius: 0 4px 4px 0;
      font-size: 0.95rem;
      color: #334155;
      margin-bottom: 1.25rem;
    }
    .demographics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
      background: #f1f5f9;
      padding: 1rem;
      border-radius: 6px;
      font-size: 0.85rem;
    }
    .demographics-item span {
      display: block;
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      margin-bottom: 0.15rem;
    }
    .demographics-item strong {
      color: var(--text);
      font-size: 0.9rem;
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 2rem 0 1rem 0;
    }
    .section-header h3 {
      font-size: 1.15rem;
      font-weight: 700;
    }
    .rubric-legend {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      font-size: 0.75rem;
    }
    .rubric-pill {
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      background: #f1f5f9;
      border: 1px solid var(--border);
    }

    .candidate-card {
      background: var(--card-bg);
      border: 1.5px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      margin-bottom: 1.25rem;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .candidate-card.graded {
      border-color: #86efac;
      background: #fcfdfc;
    }
    .candidate-card.missing-grade {
      border-color: var(--danger);
      background: #fffafa;
      box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.15);
    }
    .candidate-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.5rem;
      gap: 1rem;
    }
    .candidate-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text);
    }
    .candidate-meta {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    .rules-list {
      margin: 0.75rem 0;
      padding-left: 1.25rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .rules-list li {
      margin-bottom: 0.25rem;
    }

    .rating-container {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px dashed var(--border);
    }
    .rating-label {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      display: flex;
      justify-content: space-between;
    }
    .button-group {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }
    .grade-btn {
      padding: 0.6rem 0.5rem;
      border: 1.5px solid var(--border);
      background: #ffffff;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      color: #334155;
      text-align: center;
      transition: all 0.15s ease;
    }
    .grade-btn:hover {
      background: #f1f5f9;
      border-color: #cbd5e1;
    }
    .grade-btn[data-grade="1"].active {
      background: #0284c7;
      color: #ffffff;
      border-color: #0369a1;
      box-shadow: 0 2px 4px rgba(2, 132, 199, 0.3);
    }
    .grade-btn[data-grade="2"].active {
      background: #f59e0b;
      color: #ffffff;
      border-color: #d97706;
      box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3);
    }
    .grade-btn[data-grade="3"].active {
      background: #16a34a;
      color: #ffffff;
      border-color: #15803d;
      box-shadow: 0 2px 4px rgba(22, 163, 74, 0.3);
    }

    .justification-input {
      width: 100%;
      padding: 0.5rem 0.75rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.85rem;
      font-family: inherit;
      resize: vertical;
      min-height: 48px;
    }
    .justification-input:focus {
      outline: none;
      border-color: var(--primary);
    }

    .actions-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 2rem;
      padding: 1.25rem;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      position: sticky;
      bottom: 1rem;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
      z-index: 50;
    }
    .btn {
      padding: 0.65rem 1.25rem;
      border-radius: 6px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease;
    }
    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-secondary {
      background: #ffffff;
      border-color: var(--border);
      color: var(--text);
    }
    .btn-secondary:hover:not(:disabled) {
      background: #f1f5f9;
    }
    .btn-primary {
      background: var(--primary);
      color: #ffffff;
    }
    .btn-primary:hover:not(:disabled) {
      background: var(--primary-hover);
    }
    .btn-save {
      background: var(--success);
      color: #ffffff;
    }
    .btn-save:hover:not(:disabled) {
      background: #15803d;
    }

    .alert {
      padding: 1rem 1.25rem;
      border-radius: 6px;
      margin-bottom: 1.5rem;
      font-size: 0.9rem;
      display: none;
    }
    .alert.visible { display: block; }
    .alert-danger { background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; }
    .alert-success { background: #dcfce7; border: 1px solid #86efac; color: #166534; }
    .alert-info { background: #e0f2fe; border: 1px solid #7dd3fc; color: #0369a1; }

    @media (max-width: 768px) {
      .button-group { grid-template-columns: 1fr; }
      .actions-bar { flex-direction: column; gap: 0.75rem; }
      .actions-bar > div { width: 100%; display: flex; justify-content: space-between; }
    }
  </style>
</head>
<body>

<header>
  <div class="header-content">
    <div class="logo-area">
      <h1>SchemeIQ+ Expert Relevance Annotation</h1>
      <p>Phase 7F.1 &bull; Advisory Learning-to-Rank Benchmark Collection</p>
    </div>
    <div class="header-meta">
      <div id="annotatorBadge" class="badge badge-primary">Annotator: Loading...</div>
      <div id="progressBadge" class="badge badge-info">Archetype -/-</div>
      <div id="judgmentsBadge" class="badge badge-warning">0 Judgments</div>
    </div>
  </div>
  <div class="progress-bar-container">
    <div id="progressBar" class="progress-bar" style="width: 0%;"></div>
  </div>
</header>

<div class="container">
  <div id="alertBox" class="alert"></div>

  <section class="card" id="archetypeCard">
    <div class="archetype-title-row">
      <div>
        <span class="archetype-id-tag" id="archIdTag">ARCH---</span>
        <h2 id="archTitle">Loading Archetype Profile...</h2>
      </div>
    </div>
    <div class="narrative-box" id="archNarrative">Please wait...</div>
    <div class="demographics-grid" id="demographicsGrid"></div>
  </section>

  <div class="section-header">
    <h3>Eligible Candidates (<span id="candCount">0</span> schemes)</h3>
    <div class="rubric-legend">
      <div class="rubric-pill"><strong>[1]</strong> Broadly Relevant</div>
      <div class="rubric-pill"><strong>[2]</strong> High Interest</div>
      <div class="rubric-pill"><strong>[3]</strong> Primary Choice</div>
    </div>
  </div>

  <section id="candidatesList"></section>

  <div class="actions-bar">
    <button id="prevBtn" class="btn btn-secondary" onclick="navigate(-1)">&larr; Previous Archetype</button>
    <button id="saveDatasetBtn" class="btn btn-save" onclick="saveDataset()">&#128190; Save Dataset to Disk</button>
    <button id="nextBtn" class="btn btn-primary" onclick="submitAndNext()">Submit &amp; Next Archetype &rarr;</button>
  </div>
</div>

<script>
  let sessionState = null;
  // Local working state for active archetype's candidate selections
  const currentSelections = {};

  async function fetchSession() {
    try {
      const res = await fetch('/api/session');
      if (!res.ok) {
        throw new Error('Failed to load session');
      }
      sessionState = await res.json();
      renderSession();
    } catch (err) {
      showAlert(err.message, 'danger');
    }
  }

  function showAlert(message, type = 'info') {
    const box = document.getElementById('alertBox');
    box.className = `alert alert-${type} visible`;
    box.innerText = message;
    box.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function clearAlert() {
    const box = document.getElementById('alertBox');
    box.className = 'alert';
    box.innerText = '';
  }

  function renderSession() {
    clearAlert();
    const data = sessionState;

    // Header metadata
    document.getElementById('annotatorBadge').innerText = `Annotator: ${data.annotator_id}`;
    document.getElementById('progressBadge').innerText = data.progress_text;
    document.getElementById('judgmentsBadge').innerText = `${data.total_judgments} Judgments`;

    const progressPct = ((data.current_index + 1) / data.total_archetypes) * 100;
    document.getElementById('progressBar').style.width = `${progressPct}%`;

    // Archetype card
    const arch = data.archetype;
    document.getElementById('archIdTag').innerText = arch.archetype_id;
    document.getElementById('archTitle').innerText = arch.archetype_name;
    document.getElementById('archNarrative').innerText = arch.narrative || 'No context narrative provided.';

    const demoGrid = document.getElementById('demographicsGrid');
    demoGrid.innerHTML = '';
    const demo = arch.demographics_summary || {};
    for (const [key, val] of Object.entries(demo)) {
      const item = document.createElement('div');
      item.className = 'demographics-item';
      item.innerHTML = `<span>${key.replace(/_/g, ' ')}</span><strong>${val}</strong>`;
      demoGrid.appendChild(item);
    }

    // Candidates
    const candList = document.getElementById('candidatesList');
    candList.innerHTML = '';
    document.getElementById('candCount').innerText = data.candidates.length;

    // Reset local working selections for this archetype
    for (const key in currentSelections) delete currentSelections[key];

    data.candidates.forEach((cand, idx) => {
      // If candidate already has recorded grade in session, seed it
      currentSelections[cand.scheme_id] = {
        grade: cand.current_grade, // null if unrated
        justification: cand.current_justification || ''
      };

      const card = document.createElement('div');
      card.id = `cand-card-${cand.scheme_id}`;
      card.className = `candidate-card ${cand.current_grade ? 'graded' : ''}`;

      let rulesHtml = '';
      if (cand.matched_rules && cand.matched_rules.length > 0) {
        rulesHtml = '<ul class="rules-list">' + cand.matched_rules.map(r => `<li>✓ <strong>${r.field}</strong>: ${r.reason}</li>`).join('') + '</ul>';
      }

      const statusBadgeClass = cand.eligibility_status === 'ELIGIBLE' ? 'badge-success' : 'badge-info';

      card.innerHTML = `
        <div class="candidate-header">
          <div>
            <div class="candidate-title">[${cand.scheme_id}] ${cand.scheme_name}</div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem;">
              Scope: <strong>${cand.scheme_scope}</strong> &bull; Category: <strong>${cand.category}</strong>
            </div>
          </div>
          <div class="candidate-meta">
            <span class="badge ${statusBadgeClass}">${cand.eligibility_status}</span>
            <span class="badge badge-primary">Score: ${cand.deterministic_score}</span>
          </div>
        </div>
        ${rulesHtml}
        <div class="rating-container">
          <div class="rating-label">
            <span>Relevance Grade (Select One):</span>
            <span id="grade-status-${cand.scheme_id}" style="color: ${cand.current_grade ? 'var(--success)' : 'var(--text-muted)'}; font-weight: 700;">
              ${cand.current_grade ? '✓ Assigned: Grade ' + cand.current_grade : '⚠️ Selection Required'}
            </span>
          </div>
          <div class="button-group">
            <button type="button" class="grade-btn ${cand.current_grade === 1 ? 'active' : ''}" data-grade="1" onclick="selectGrade('${cand.scheme_id}', 1)">
              1 &mdash; Broadly Relevant
            </button>
            <button type="button" class="grade-btn ${cand.current_grade === 2 ? 'active' : ''}" data-grade="2" onclick="selectGrade('${cand.scheme_id}', 2)">
              2 &mdash; High Interest
            </button>
            <button type="button" class="grade-btn ${cand.current_grade === 3 ? 'active' : ''}" data-grade="3" onclick="selectGrade('${cand.scheme_id}', 3)">
              3 &mdash; Primary Choice
            </button>
          </div>
          <textarea 
            class="justification-input" 
            placeholder="Optional expert justification or notes for this grade..." 
            oninput="updateJustification('${cand.scheme_id}', this.value)"
          >${cand.current_justification || ''}</textarea>
        </div>
      `;
      candList.appendChild(card);
    });

    // Navigation buttons state
    document.getElementById('prevBtn').disabled = (data.current_index === 0);
    if (data.current_index === data.total_archetypes - 1) {
      document.getElementById('nextBtn').innerHTML = 'Submit &amp; Finish Archetypes &rarr;';
    } else {
      document.getElementById('nextBtn').innerHTML = 'Submit &amp; Next Archetype &rarr;';
    }
  }

  function selectGrade(schemeId, grade) {
    if (!currentSelections[schemeId]) {
      currentSelections[schemeId] = { grade: null, justification: '' };
    }
    currentSelections[schemeId].grade = grade;

    const card = document.getElementById(`cand-card-${schemeId}`);
    card.classList.remove('missing-grade');
    card.classList.add('graded');

    const buttons = card.querySelectorAll('.grade-btn');
    buttons.forEach(btn => {
      if (parseInt(btn.getAttribute('data-grade'), 10) === grade) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    const statusSpan = document.getElementById(`grade-status-${schemeId}`);
    const labels = {1: '1 (Broadly Relevant)', 2: '2 (High Interest)', 3: '3 (Primary Choice)'};
    statusSpan.style.color = 'var(--success)';
    statusSpan.innerText = `✓ Assigned: Grade ${labels[grade]}`;
  }

  function updateJustification(schemeId, text) {
    if (!currentSelections[schemeId]) {
      currentSelections[schemeId] = { grade: null, justification: '' };
    }
    currentSelections[schemeId].justification = text;
  }

  async function submitAndNext() {
    clearAlert();
    const data = sessionState;
    const candidates = data.candidates;

    // Requirement 8: Prevent submission without a relevance grade
    let firstMissingCard = null;
    let missingCount = 0;

    for (const cand of candidates) {
      const sel = currentSelections[cand.scheme_id];
      const card = document.getElementById(`cand-card-${cand.scheme_id}`);
      if (!sel || sel.grade === null || sel.grade === undefined) {
        missingCount++;
        card.classList.add('missing-grade');
        if (!firstMissingCard) firstMissingCard = card;
      } else {
        card.classList.remove('missing-grade');
      }
    }

    if (missingCount > 0) {
      showAlert(`Cannot proceed: ${missingCount} candidate scheme(s) are missing a relevance grade. All eligible candidates must have an explicit grade (1, 2, or 3).`, 'danger');
      if (firstMissingCard) {
        firstMissingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      return;
    }

    // Build payload
    const judgments = candidates.map(cand => ({
      scheme_id: cand.scheme_id,
      grade: currentSelections[cand.scheme_id].grade,
      justification: currentSelections[cand.scheme_id].justification
    }));

    const nextIndex = (data.current_index < data.total_archetypes - 1) ? data.current_index + 1 : data.current_index;

    try {
      const res = await fetch('/api/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          archetype_id: data.archetype.archetype_id,
          judgments: judgments,
          next_index: nextIndex
        })
      });

      const respJson = await res.json();
      if (!res.ok) {
        throw new Error(respJson.error || 'Submission failed');
      }

      // If at last archetype, notify
      if (data.current_index === data.total_archetypes - 1) {
        showAlert('All archetypes have been graded! Click "Save Dataset to Disk" to validate and export your judgments.', 'success');
        await fetchSession();
      } else {
        await fetchSession();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    } catch (err) {
      showAlert(err.message, 'danger');
    }
  }

  async function navigate(direction) {
    if (!sessionState) return;
    const target = sessionState.current_index + direction;
    if (target < 0 || target >= sessionState.total_archetypes) return;

    try {
      const res = await fetch('/api/navigate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_index: target })
      });
      const respJson = await res.json();
      if (!res.ok) {
        throw new Error(respJson.error || 'Navigation failed');
      }
      await fetchSession();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      showAlert(err.message, 'danger');
    }
  }

  async function saveDataset() {
    clearAlert();
    try {
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const resp = await res.json();
      if (!res.ok) {
        throw new Error(resp.error || 'Failed to save dataset');
      }

      const dist = JSON.stringify(resp.label_distribution || {});
      showAlert(`✓ Success! Dataset validated and saved to: ${resp.output_path} (Total Records: ${resp.total_records}, Groups: ${resp.total_groups}, Label Distribution: ${dist})`, 'success');
    } catch (err) {
      showAlert(err.message, 'danger');
    }
  }

  // Initial load
  window.addEventListener('DOMContentLoaded', fetchSession);
</script>
</body>
</html>
"""


def create_app(
    archetypes_path: Optional[Path] = None,
    annotator_id: str = "expert_01",
    output_path: Optional[Path] = None,
    evaluator: Optional[SchemeEligibilityEvaluator] = None,
) -> Flask:
    """
    Flask Application Factory for SchemeIQ+ Web Annotation Interface.
    Reuses existing ExpertAnnotationSession and SchemeEligibilityEvaluator.
    """
    app = Flask(__name__)

    # Default paths
    real_arch_path = archetypes_path or Path("data/ranking/real_archetypes.json")
    clean_annotator_id = validate_annotator_id(annotator_id)
    default_out_path = output_path or Path(f"data/ranking/annotations_{clean_annotator_id}.json")

    # Load archetypes
    archetypes: List[CitizenArchetype] = load_archetypes_from_file(real_arch_path)
    if not archetypes:
        raise ValueError(f"No citizen archetypes found in {real_arch_path}")

    # Shared session instance
    session = ExpertAnnotationSession(
        annotator_id=clean_annotator_id,
        evaluator=evaluator or SchemeEligibilityEvaluator(),
    )

    # In-memory application state
    app_state: Dict[str, Any] = {
        "archetypes": archetypes,
        "session": session,
        "current_index": 0,
        "output_path": default_out_path,
        "archetypes_path": real_arch_path,
    }
    app.config["ANNOTATION_STATE"] = app_state

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/session", methods=["GET"])
    def get_session_state():
        state = app.config["ANNOTATION_STATE"]
        archs: List[CitizenArchetype] = state["archetypes"]
        sess: ExpertAnnotationSession = state["session"]
        curr_idx: int = state["current_index"]

        if curr_idx < 0 or curr_idx >= len(archs):
            curr_idx = 0
            state["current_index"] = 0

        arch = archs[curr_idx]
        candidate_evals = sess.get_candidate_schemes(arch)

        # Existing judgments in session for this archetype
        existing_judgments = {
            r.scheme_id: (r.relevance_label, r.notes)
            for r in sess.records
            if r.query_id == arch.archetype_id
        }

        candidates_payload = []
        for cand in candidate_evals:
            existing = existing_judgments.get(cand.scheme_id)
            candidates_payload.append({
                "scheme_id": cand.scheme_id,
                "scheme_name": cand.scheme_name,
                "category": cand.category,
                "scheme_scope": cand.scheme_scope,
                "eligibility_status": cand.eligibility_status.value,
                "deterministic_score": round(cand.score, 2),
                "matched_rules": [
                    {"field": mr.field, "reason": mr.reason}
                    for mr in cand.matched_rules
                ],
                # Strict zero auto-inference: null if not explicitly assigned by expert
                "current_grade": existing[0] if existing else None,
                "current_justification": existing[1] if existing else None,
            })

        payload = {
            "annotator_id": sess.annotator_id,
            "current_index": curr_idx,
            "total_archetypes": len(archs),
            "progress_text": f"Archetype {curr_idx + 1} of {len(archs)}",
            "total_judgments": len(sess.records),
            "output_path": str(state["output_path"]),
            "archetype": {
                "archetype_id": arch.archetype_id,
                "archetype_name": arch.archetype_name,
                "narrative": arch.narrative,
                "demographics_summary": arch.profile.model_dump(exclude_none=True),
            },
            "candidates": candidates_payload,
        }
        return jsonify(payload), 200

    @app.route("/api/annotator", methods=["POST"])
    def update_annotator():
        state = app.config["ANNOTATION_STATE"]
        data = request.get_json(silent=True) or {}
        new_annotator_id = data.get("annotator_id")

        try:
            clean_id = validate_annotator_id(new_annotator_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Update session annotator ID and adjust default output path
        state["session"].annotator_id = clean_id
        state["output_path"] = Path(f"data/ranking/annotations_{clean_id}.json")
        return jsonify({"status": "ok", "annotator_id": clean_id}), 200

    @app.route("/api/submit", methods=["POST"])
    def submit_judgments():
        state = app.config["ANNOTATION_STATE"]
        archs: List[CitizenArchetype] = state["archetypes"]
        sess: ExpertAnnotationSession = state["session"]
        data = request.get_json(silent=True) or {}

        arch_id = data.get("archetype_id")
        judgments = data.get("judgments")
        next_index = data.get("next_index")

        if not arch_id:
            return jsonify({"error": "Missing 'archetype_id' in submission"}), 400

        if not isinstance(judgments, list):
            return jsonify({"error": "Expected 'judgments' list in submission"}), 400

        arch = next((a for a in archs if a.archetype_id == arch_id), None)
        if not arch:
            return jsonify({"error": f"Archetype '{arch_id}' not found"}), 400

        # Retrieve strictly eligible candidate schemes
        candidate_evals = sess.get_candidate_schemes(arch)
        candidate_map = {c.scheme_id: c for c in candidate_evals}

        # Validate that EVERY eligible candidate has an explicit grade in judgments
        submitted_schemes = {}
        for j in judgments:
            if not isinstance(j, dict):
                return jsonify({"error": "Each judgment must be an object"}), 400
            sid = j.get("scheme_id")
            grade = j.get("grade")
            justification = j.get("justification")
            if not sid:
                return jsonify({"error": "Missing 'scheme_id' in judgment entry"}), 400
            submitted_schemes[sid] = (grade, justification)

        # Requirement 8: Prevent submission without a relevance grade
        for cand_id, cand_eval in candidate_map.items():
            if cand_id not in submitted_schemes or submitted_schemes[cand_id][0] is None:
                return jsonify({
                    "error": f"Missing relevance grade for candidate scheme '{cand_id}'. "
                             "All eligible candidates must be assigned an explicit grade in {1, 2, 3}."
                }), 400

            grade, justification = submitted_schemes[cand_id]
            if grade not in VALID_EXPERT_RELEVANCE_GRADES:
                return jsonify({
                    "error": f"Invalid relevance grade '{grade}' for scheme '{cand_id}'. Must be 1, 2, or 3."
                }), 400

        # Disallow grading schemes not in candidate set
        for sid in submitted_schemes:
            if sid not in candidate_map:
                return jsonify({
                    "error": f"Scheme '{sid}' is not an eligible candidate for archetype '{arch_id}'."
                }), 400

        # Record or update judgments cleanly (prevent duplicates)
        for cand_id, (grade, justification) in submitted_schemes.items():
            cand_eval = candidate_map[cand_id]
            sess.record_or_update_judgment(
                archetype=arch,
                candidate_eval=cand_eval,
                relevance_label=int(grade),
                justification=justification,
            )

        # Handle index progression
        if next_index is not None and isinstance(next_index, int):
            if 0 <= next_index < len(archs):
                state["current_index"] = next_index

        return jsonify({
            "status": "ok",
            "current_index": state["current_index"],
            "total_records": len(sess.records),
        }), 200

    @app.route("/api/navigate", methods=["POST"])
    def navigate():
        state = app.config["ANNOTATION_STATE"]
        archs: List[CitizenArchetype] = state["archetypes"]
        data = request.get_json(silent=True) or {}
        target_index = data.get("target_index")

        if target_index is None or not isinstance(target_index, int):
            return jsonify({"error": "Missing or invalid 'target_index'"}), 400

        if target_index < 0 or target_index >= len(archs):
            return jsonify({"error": f"Target index {target_index} out of range [0, {len(archs)-1}]"}), 400

        state["current_index"] = target_index
        return jsonify({"status": "ok", "current_index": target_index}), 200

    @app.route("/api/save", methods=["POST"])
    def save_dataset():
        state = app.config["ANNOTATION_STATE"]
        sess: ExpertAnnotationSession = state["session"]
        data = request.get_json(silent=True) or {}

        target_path_str = data.get("output_path")
        target_path = Path(target_path_str) if target_path_str else state["output_path"]

        if len(sess.records) == 0:
            return jsonify({"error": "No judgments recorded yet. Cannot save an empty dataset."}), 400

        try:
            # Build and validate using existing RankingDatasetValidator
            dataset = sess.build_dataset(dataset_name=f"SchemeIQ_Expert_Annotations_{sess.annotator_id}")
            report = RankingDatasetValidator.validate_dataset(dataset)
            if not report.is_valid:
                return jsonify({"error": f"Validation failed: {report.errors}"}), 400

            saved_path = RankingDatasetIO.save_json(dataset, target_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save dataset: {str(e)}"}), 500

        return jsonify({
            "status": "ok",
            "output_path": str(saved_path),
            "total_records": len(dataset.records),
            "total_groups": dataset.metadata.total_groups,
            "label_distribution": dataset.metadata.label_distribution,
            "validator_passed": True,
        }), 200

    @app.route("/api/summary", methods=["GET"])
    def get_summary():
        state = app.config["ANNOTATION_STATE"]
        sess: ExpertAnnotationSession = state["session"]
        archs: List[CitizenArchetype] = state["archetypes"]

        label_dist: Dict[str, int] = {"1": 0, "2": 0, "3": 0}
        for r in sess.records:
            k = str(r.relevance_label)
            label_dist[k] = label_dist.get(k, 0) + 1

        return jsonify({
            "annotator_id": sess.annotator_id,
            "current_index": state["current_index"],
            "total_archetypes": len(archs),
            "total_records": len(sess.records),
            "label_distribution": label_dist,
            "records": [
                {
                    "query_id": r.query_id,
                    "scheme_id": r.scheme_id,
                    "relevance_label": r.relevance_label,
                    "notes": r.notes,
                }
                for r in sess.records
            ],
        }), 200

    return app


def launch_web_annotation(
    archetypes_path: Path,
    annotator_id: str,
    output_path: Path,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> None:
    """
    Launch local web annotation server.
    """
    print("=" * 75)
    print("  SchemeIQ+ Web Annotation Interface (Phase 7F.1)")
    print(f"  Annotator ID: {annotator_id}")
    print(f"  Input File:   {archetypes_path}")
    print(f"  Output File:  {output_path}")
    print(f"  Local URL:    http://{host}:{port}")
    print("=" * 75)
    app = create_app(
        archetypes_path=archetypes_path,
        annotator_id=annotator_id,
        output_path=output_path,
    )
    app.run(host=host, port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="SchemeIQ+ Simple Web Annotation UI (Phase 7F.1)")
    parser.add_argument("--input", default="data/ranking/real_archetypes.json", help="Path to archetypes JSON file")
    parser.add_argument("--annotator", default="expert_01", help="Non-PII expert annotator ID (e.g. 'expert_01')")
    parser.add_argument("--output", default="data/ranking/expert_annotations.json", help="Output path for validated dataset")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind the web server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the web server (default: 5000)")

    args = parser.parse_args()
    launch_web_annotation(
        archetypes_path=Path(args.input),
        annotator_id=args.annotator,
        output_path=Path(args.output),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
