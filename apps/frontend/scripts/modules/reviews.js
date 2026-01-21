import { el } from './dom.js';
import { state } from './state.js';
import { utils } from './utils.js';
import { lang } from './language.js';
import { API } from './api.js';
import { ui } from './ui.js';

// Review Management Factory
const createReviewManager = (type, elements) => ({
  hide() {
    state.pendingReviews[type] = null;
    if (elements.body) {
      elements.body.innerHTML = '<p class="review-instructions">Review and adjust each segment before continuing.</p>';
    }
    if (elements.status) {
      elements.status.textContent = `Pipeline paused for ${type} review.`;
    }
    [elements.apply, elements.skip, elements.add].forEach(btn => {
      if (btn) {
        btn.disabled = false;
        if (btn === elements.add) btn.hidden = true;
      }
    });
    if (elements.card) elements.card.hidden = true;
  },

  show(event) {
    // Implementation delegated to specific review types
  },

  async submit(applyChanges) {
    // Implementation delegated to specific review types
  }
});

// Transcription Review
const transcriptionReview = {
  ...createReviewManager("transcription", el.transcription),

  show(event) {
    if (!el.transcription.card || !el.transcription.body) return;
    
    const rawPayload = event?.raw || {};
    const runId = event?.run_id || "";
    const duration = typeof event?.duration === "number" ? event.duration : null;
    const tolerance = Number.isFinite(event?.tolerance) ? event.tolerance : 0.25;
    
    let supportedLangs = Array.isArray(event?.languages)
      ? [...new Set(event.languages.map(l => String(l).trim().toLowerCase()).filter(Boolean))]
      : [];
    
    supportedLangs.forEach(code => lang.register(code));
    supportedLangs.sort((a, b) => 
      (lang.toLabel(a) || a).localeCompare(lang.toLabel(b) || b)
    );
    
    state.pendingReviews.transcription = {
      runId,
      payload: utils.clone(rawPayload),
      duration,
      languages: supportedLangs,
      tolerance
    };
    
    const review = state.pendingReviews.transcription;
    if (!Array.isArray(review.payload.segments)) {
      review.payload.segments = [];
    }
    
    review.payload.segments.forEach(seg => {
      if (!seg.segment_id) seg.segment_id = utils.generateId();
      seg.start = Number.isFinite(Number(seg.start)) ? Number(seg.start) : 0;
      seg.end = Number.isFinite(Number(seg.end)) ? Number(seg.end) : seg.start;
      seg.lang = seg.lang || "";
      seg.text = seg.text || "";
      if (seg.lang) lang.register(seg.lang);
    });
    
    this.render();
    
    if (el.transcription.status) {
      const durationLabel = Number.isFinite(duration)
        ? ` Audio length: ${duration.toFixed(3)}s (±${tolerance.toFixed(2)} s tolerance).`
        : "";
      el.transcription.status.textContent = `Pipeline paused for transcription review.${durationLabel}`;
    }
    
    [el.transcription.apply, el.transcription.skip, el.transcription.add].forEach(btn => {
      if (btn) {
        btn.disabled = false;
        if (btn === el.transcription.add) btn.hidden = false;
      }
    });
    
    el.transcription.card.hidden = false;
    el.transcription.card.scrollIntoView({ behavior: "smooth", block: "center" });
  },

  render() {
    const review = state.pendingReviews.transcription;
    if (!review || !el.transcription.body) return;
    
    const { payload, duration, languages, tolerance = 0 } = review;
    const segments = payload.segments || [];
    
    el.transcription.body.innerHTML = "";
    
    const intro = document.createElement("p");
    intro.className = "review-instructions";
    const durationLabel = Number.isFinite(duration)
      ? `Total audio duration: ${duration.toFixed(3)}s (±${tolerance.toFixed(2)} s tolerance).`
      : "Adjust segment timings, languages, and text as needed.";
    intro.textContent = `Adjust segment timings, languages, and text. ${durationLabel}`;
    el.transcription.body.appendChild(intro);
    
    if (!segments.length) {
      const empty = document.createElement("p");
      empty.className = "review-instructions";
      empty.textContent = "No segments available. Use 'Add segment' to create one.";
      el.transcription.body.appendChild(empty);
      return;
    }
    
    segments.forEach((seg, i) => this.renderSegment(seg, i, { duration, tolerance, languages }));
  },

  renderSegment(seg, index, { duration, tolerance, languages }) {
    const wrapper = document.createElement("div");
    wrapper.className = "review-segment";
    wrapper.dataset.segmentId = seg.segment_id || "";
    wrapper.dataset.index = String(index);
    
    const startVal = Number(seg.start) || 0;
    const endVal = Number(seg.end) || 0;
    const langDisplay = seg.lang ? ` · ${lang.toLabel(seg.lang) || seg.lang}` : "";
    
    const meta = document.createElement("div");
    meta.className = "segment-meta";
    meta.textContent = `Segment ${index + 1} (${utils.formatTime(startVal)} — ${utils.formatTime(endVal)} · ${startVal.toFixed(3)}s — ${endVal.toFixed(3)}s)${langDisplay}`;
    
    const createInput = (labelText, type, value, attrs = {}) => {
      const label = document.createElement("label");
      label.textContent = labelText;
      const input = document.createElement(type === "textarea" ? "textarea" : "input");
      if (type !== "textarea") input.type = type;
      Object.entries(attrs).forEach(([k, v]) => {
        if (k.startsWith("data-")) input.dataset[k.slice(5)] = v;
        else input.setAttribute(k, v);
      });
      input.value = value;
      label.appendChild(input);
      return label;
    };
    
    const maxTime = Number.isFinite(duration) && duration > 0 
      ? String((duration + tolerance).toFixed(3)) 
      : undefined;
    
    const startLabel = createInput("Start (s)", "number", startVal, {
      step: "0.001",
      min: "0",
      max: maxTime,
      "data-role": "segment-start"
    });
    
    const endLabel = createInput("End (s)", "number", endVal, {
      step: "0.001",
      min: "0",
      max: maxTime,
      "data-role": "segment-end"
    });
    
    const langLabel = document.createElement("label");
    langLabel.textContent = "Language";
    let langControl;
    
    if (languages?.length) {
      langControl = document.createElement("select");
      langControl.dataset.role = "segment-lang";
      langControl.appendChild(new Option("Auto", ""));
      languages.forEach(code => {
        langControl.appendChild(new Option(lang.toLabel(code) || code, code));
      });
      const normalized = (seg.lang || "").toLowerCase();
      langControl.value = languages.includes(normalized) ? normalized : "";
    } else {
      langControl = document.createElement("input");
      langControl.type = "text";
      langControl.placeholder = "auto";
      langControl.dataset.role = "segment-lang";
      langControl.setAttribute("list", "language-list");
      langControl.value = seg.lang || "";
    }
    langLabel.appendChild(langControl);
    
    const textLabel = createInput("Text", "textarea", seg.text || "", {
      "data-role": "segment-text",
      rows: Math.max(2, Math.min(6, Math.ceil((seg.text?.length || 1) / 60)))
    });
    
    const actions = document.createElement("div");
    actions.className = "segment-actions";
    
    const createBtn = (text, className, handler) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = className;
      btn.textContent = text;
      btn.onclick = handler;
      return btn;
    };
    
    actions.append(
      createBtn("Add before", "secondary", () => this.addSegment(index - 1)),
      createBtn("Add after", "secondary", () => this.addSegment(index)),
      createBtn("Delete", "ghost", () => this.removeSegment(index))
    );
    
    wrapper.append(meta, startLabel, endLabel, langLabel, textLabel, actions);
    el.transcription.body.appendChild(wrapper);
  },

  addSegment(afterIndex) {
    const review = state.pendingReviews.transcription;
    if (!review) return;
    
    const segments = review.payload.segments || [];
    const { duration, tolerance = 0 } = review;
    
    const prev = afterIndex >= 0 ? segments[afterIndex] : null;
    const next = segments[afterIndex + 1] || null;
    
    const prevEnd = prev ? Number(prev.end ?? prev.start ?? 0) : 0;
    const nextStart = next ? Number(next.start ?? prevEnd) : duration;
    const gap = Number.isFinite(nextStart) ? Math.max(0, nextStart - prevEnd) : 1;
    
    let start = Number.isFinite(prevEnd) ? prevEnd : 0;
    if (duration !== null) start = Math.min(Math.max(0, start), duration + tolerance);
    
    let end = start + (gap > 0 ? Math.min(1, gap) : 1);
    if (duration !== null) end = Math.min(end, duration + tolerance);
    if (end <= start) end = duration !== null ? Math.min(duration + tolerance, start + 0.5) : start + 0.5;
    
    segments.splice(Math.max(afterIndex + 1, 0), 0, {
      segment_id: utils.generateId(),
      start: Number.parseFloat(start.toFixed(3)),
      end: Number.parseFloat(end.toFixed(3)),
      text: "",
      lang: "",
      speaker_id: null,
      words: null
    });
    
    this.render();
  },

  removeSegment(index) {
    const review = state.pendingReviews.transcription;
    if (!review) return;
    
    const segments = review.payload.segments || [];
    if (index < 0 || index >= segments.length) return;
    
    segments.splice(index, 1);
    this.render();
  },

  async submit(applyChanges) {
    const review = state.pendingReviews.transcription;
    if (!review?.runId) return false;
    
    const submission = utils.clone(review.payload);
    if (!Array.isArray(submission.segments)) submission.segments = [];
    
    if (applyChanges && el.transcription.body) {
      const { payload, duration, tolerance = 0 } = review;
      const baseSegments = payload.segments || [];
      const updatedSegments = [];
      const errors = new Set();
      let hasError = false;
      
      el.transcription.body.querySelectorAll(".review-segment").forEach(segEl => {
        const segId = segEl.dataset.segmentId || utils.generateId();
        const startInput = segEl.querySelector('[data-role="segment-start"]');
        const endInput = segEl.querySelector('[data-role="segment-end"]');
        const langControl = segEl.querySelector('[data-role="segment-lang"]');
        const textArea = segEl.querySelector('[data-role="segment-text"]');
        
        const startVal = Number.parseFloat(startInput?.value);
        const endVal = Number.parseFloat(endInput?.value ?? startVal);
        
        const startValid = Number.isFinite(startVal) && startVal >= -tolerance &&
          (duration === null || startVal <= duration + tolerance);
        const endValid = Number.isFinite(endVal) && endVal >= -tolerance &&
          (duration === null || endVal <= duration + tolerance) &&
          Number.isFinite(startVal) && endVal >= startVal;
        
        let segError = false;
        if (!startValid) {
          hasError = segError = true;
          errors.add(`Segment start times must stay within the audio duration (±${tolerance.toFixed(2)} s).`);
        }
        if (!endValid) {
          hasError = segError = true;
          errors.add(`Segment end times must stay within the audio duration (±${tolerance.toFixed(2)} s) and ≥ start.`);
        }
        
        utils.markValidity(startInput, startValid, `Start must be within audio duration (±${tolerance.toFixed(2)} s).`);
        utils.markValidity(endInput, endValid, `End must be within audio duration (±${tolerance.toFixed(2)} s) and ≥ start.`);
        
        if (!segError) {
          const clampedStart = duration === null ? Math.max(0, startVal) :
            Math.min(Math.max(0, startVal), duration + tolerance);
          const clampedEnd = duration === null ? Math.max(clampedStart, endVal) :
            Math.min(Math.max(clampedStart, endVal), duration + tolerance);
          
          const textVal = textArea?.value.trim() || "";
          const langRaw = langControl?.value.toString().trim() || "";
          const resolvedLang = lang.resolve(langRaw) || langRaw;
          const langVal = resolvedLang ? resolvedLang.toLowerCase() : "";
          
          const reference = baseSegments.find(s => s.segment_id === segId) || {};
          
          updatedSegments.push({
            segment_id: segId,
            start: Number.parseFloat((clampedStart || 0).toFixed(3)),
            end: Number.parseFloat((clampedEnd || clampedStart || 0).toFixed(3)),
            text: textVal,
            lang: langVal,
            speaker_id: reference.speaker_id || null,
            words: null
          });
        }
      });
      
      if (hasError) {
        if (el.transcription.status) {
          el.transcription.status.textContent = `Fix highlighted segment issues: ${[...errors].join(" ")}`;
        }
        [el.transcription.apply, el.transcription.skip, el.transcription.add].forEach(btn => {
          if (btn) btn.disabled = false;
        });
        return false;
      }
      
      updatedSegments.sort((a, b) => a.start - b.start);
      submission.segments = updatedSegments;
    }
    
    [el.transcription.apply, el.transcription.skip, el.transcription.add].forEach(btn => {
      if (btn) btn.disabled = true;
    });
    if (el.transcription.status) el.transcription.status.textContent = "Submitting review…";
    
    try {
      const response = await fetch(API.routes.reviewTranscription, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: review.runId, transcription: submission })
      });
      
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      
      ui.log(applyChanges ? "📝 Submitted transcription corrections." : "👍 Approved transcription without changes.");
      this.hide();
      ui.setStatus("Running", "running");
      return true;
    } catch (err) {
      if (el.transcription.status) {
        el.transcription.status.textContent = `Failed to submit review: ${err.message}`;
      }
      [el.transcription.apply, el.transcription.skip, el.transcription.add].forEach(btn => {
        if (btn) btn.disabled = false;
      });
      ui.log(`❌ Failed to submit transcription review: ${err.message}`);
      return false;
    }
  }
};

// Alignment Review
const alignmentReview = {
  ...createReviewManager("alignment", el.alignment),

  collectSpeakers(segments) {
    return [...new Set((segments || []).map(s => s?.speaker_id).filter(Boolean))].sort();
  },

  normalizeSpeakers(...lists) {
    return [...new Set(lists.flat().flatMap(v =>
      Array.isArray(v) ? v : [v]
    ).map(v => String(v).trim()).filter(Boolean))].sort();
  },

  updateSpeakerOptions(speakers) {
    if (!el.alignment.speakers) return;
    el.alignment.speakers.innerHTML = "";
    (speakers || []).forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      el.alignment.speakers.appendChild(opt);
    });
  },

  refreshSpeakers(extra = []) {
    const review = state.pendingReviews.alignment;
    if (!review) return;
    
    const fromSegs = this.collectSpeakers(review.payload?.segments);
    const current = review.speakers || [];
    const normalized = this.normalizeSpeakers(current, fromSegs, extra);
    
    review.speakers = normalized;
    this.updateSpeakerOptions(normalized);
  },

  addSpeaker(name) {
    const value = (name || "").trim();
    if (!value) return;
    this.refreshSpeakers([value]);
    this.render();
  },

  hide() {
    state.pendingReviews.alignment = null;
    if (el.alignment.body) {
      el.alignment.body.innerHTML = '<p class="review-instructions">Adjust segment timings, text, and speakers. You can add, delete, or merge segments as needed.</p>';
    }
    if (el.alignment.status) {
      el.alignment.status.textContent = "Pipeline paused for alignment review.";
    }
    [el.alignment.apply, el.alignment.skip, el.alignment.add].forEach(btn => {
      if (btn) btn.disabled = false;
    });
    if (el.alignment.card) el.alignment.card.hidden = true;
    if (el.alignment.speakers) el.alignment.speakers.innerHTML = "";
  },

  show(event) {
    if (!el.alignment.card || !el.alignment.body) return;
    
    const alignedPayload = event?.aligned || {};
    const speakers = Array.isArray(event?.speakers) 
      ? event.speakers 
      : this.collectSpeakers(alignedPayload.segments);
    
    state.pendingReviews.alignment = {
      runId: event?.run_id || "",
      payload: utils.clone(alignedPayload),
      original: utils.clone(alignedPayload),
      speakers: [...new Set(speakers)].sort()
    };
    
    const review = state.pendingReviews.alignment;
    if (!Array.isArray(review.payload.segments)) review.payload.segments = [];
    if (!Array.isArray(review.original.segments)) review.original.segments = [];
    
    this.refreshSpeakers();
    this.render();
    
    if (el.alignment.status) {
      el.alignment.status.textContent = "Pipeline paused for alignment review.";
    }
    [el.alignment.apply, el.alignment.skip, el.alignment.add].forEach(btn => {
      if (btn) btn.disabled = false;
    });
    
    el.alignment.card.hidden = false;
    el.alignment.card.scrollIntoView({ behavior: "smooth", block: "center" });
  },

  render() {
    const review = state.pendingReviews.alignment;
    if (!review || !el.alignment.body) return;
    
    const segments = review.payload.segments || [];
    el.alignment.body.innerHTML = "";
    
    const intro = document.createElement("p");
    intro.className = "review-instructions";
    intro.textContent = "Adjust segment timings, text, and speakers. You can add, delete, or merge segments as needed.";
    el.alignment.body.appendChild(intro);
    
    const speakerControls = document.createElement("div");
    speakerControls.className = "review-speaker-controls";
    
    const speakerInput = document.createElement("input");
    speakerInput.type = "text";
    speakerInput.placeholder = "Add new speaker label";
    speakerInput.setAttribute("list", "speaker-list");
    
    const speakerBtn = document.createElement("button");
    speakerBtn.type = "button";
    speakerBtn.className = "secondary";
    speakerBtn.textContent = "Add speaker";
    
    const addSpeaker = () => {
      this.addSpeaker(speakerInput.value);
      speakerInput.value = "";
    };
    
    speakerBtn.onclick = addSpeaker;
    speakerInput.onkeydown = e => {
      if (e.key === "Enter") {
        e.preventDefault();
        addSpeaker();
      }
    };
    
    speakerControls.append(speakerInput, speakerBtn);
    el.alignment.body.appendChild(speakerControls);
    
    if (!segments.length) {
      const notice = document.createElement("p");
      notice.className = "review-instructions";
      notice.textContent = "No aligned segments available. Add segments below if needed.";
      el.alignment.body.appendChild(notice);
      return;
    }
    
    segments.forEach((seg, i) => this.renderSegment(seg, i));
  },

  renderSegment(seg, index) {
    const review = state.pendingReviews.alignment;
    const segments = review.payload.segments;
    
    const wrapper = document.createElement("div");
    wrapper.className = "review-segment alignment";
    wrapper.dataset.index = String(index);
    
    const meta = document.createElement("div");
    meta.className = "segment-meta";
    meta.textContent = `Segment ${index + 1} (${utils.formatTime(seg.start)} — ${utils.formatTime(seg.end)})`;
    
    const timingRow = document.createElement("div");
    timingRow.className = "flex";
    
    const createNumberInput = (labelText, value, field) => {
      const label = document.createElement("label");
      label.textContent = labelText;
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.01";
      input.value = value ?? 0;
      input.onchange = () => {
        this.updateTime(index, field, Number(input.value), meta);
      };
      label.appendChild(input);
      return label;
    };
    
    const speakerLabel = document.createElement("label");
    speakerLabel.textContent = "Speaker";
    const speakerSelect = document.createElement("select");
    speakerSelect.className = "speaker-select";
    speakerSelect.appendChild(new Option("Unassigned", ""));
    
    const knownSpeakers = review.speakers || [];
    knownSpeakers.forEach(name => speakerSelect.appendChild(new Option(name, name)));
    
    if (seg.speaker_id && !knownSpeakers.includes(seg.speaker_id)) {
      speakerSelect.appendChild(new Option(seg.speaker_id, seg.speaker_id));
    }
    
    speakerSelect.value = seg.speaker_id || "";
    speakerSelect.onchange = () => this.updateSpeaker(index, speakerSelect.value);
    speakerLabel.appendChild(speakerSelect);
    
    timingRow.append(
      createNumberInput("Start", seg.start, "start"),
      createNumberInput("End", seg.end, "end"),
      speakerLabel
    );
    
    const textArea = document.createElement("textarea");
    textArea.value = seg.text || "";
    textArea.oninput = () => this.updateText(index, textArea.value);
    
    const controls = document.createElement("div");
    controls.className = "review-actions";
    
    const createBtn = (text, handler, disabled = false) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "secondary";
      btn.textContent = text;
      btn.disabled = disabled;
      btn.onclick = handler;
      return btn;
    };
    
    controls.append(
      createBtn("Delete segment", () => this.removeSegment(index), segments.length <= 1),
      createBtn("Merge with previous", () => this.mergeSegment(index - 1), index === 0),
      createBtn("Merge with next", () => this.mergeSegment(index), index >= segments.length - 1),
      createBtn("Add after", () => this.addSegment(index))
    );
    
    wrapper.append(meta, timingRow, textArea, controls);
    el.alignment.body.appendChild(wrapper);
  },

  addSegment(afterIndex) {
    const review = state.pendingReviews.alignment;
    if (!review) {
      review = state.pendingReviews.alignment = {
        runId: "",
        payload: { segments: [] },
        original: { segments: [] },
        speakers: []
      };
    }
    
    const segments = review.payload.segments;
    const insertIndex = Number.isInteger(afterIndex) && afterIndex >= -1 ? afterIndex + 1 : segments.length;
    const previous = afterIndex >= 0 ? segments[afterIndex] : segments[segments.length - 1];
    const next = segments[afterIndex + 1];
    
    const baseStart = typeof previous?.end === "number" ? previous.end :
      typeof previous?.start === "number" ? previous.start : 0;
    const nextStart = typeof next?.start === "number" ? next.start : baseStart + 1;
    
    const speakerList = review.speakers || [];
    const defaultSpeaker = segments[afterIndex]?.speaker_id ||
      segments[Math.max(0, segments.length - 1)]?.speaker_id ||
      speakerList[speakerList.length - 1] || speakerList[0] || "";
    
    segments.splice(insertIndex, 0, {
      start: baseStart,
      end: Math.max(baseStart, nextStart),
      text: "",
      speaker_id: defaultSpeaker
    });
    
    this.refreshSpeakers([defaultSpeaker]);
    this.render();
  },

  removeSegment(index) {
    const review = state.pendingReviews.alignment;
    if (!review) return;
    
    const segments = review.payload.segments;
    if (!Array.isArray(segments) || segments.length <= 1) return;
    
    segments.splice(index, 1);
    this.refreshSpeakers();
    this.render();
  },

  mergeSegment(index) {
    const review = state.pendingReviews.alignment;
    if (!review) return;
    
    const segments = review.payload.segments;
    if (!Array.isArray(segments) || index < 0 || index >= segments.length - 1) return;
    
    const current = segments[index];
    const next = segments[index + 1];
    
    current.text = [current.text || "", next.text || ""].filter(Boolean).join(" ").trim();
    if (typeof next.end === "number") current.end = next.end;
    if (!current.speaker_id && next.speaker_id) current.speaker_id = next.speaker_id;
    current.words = null;
    
    segments.splice(index + 1, 1);
    this.refreshSpeakers();
    this.render();
  },

  updateTime(index, field, value, metaEl) {
    const review = state.pendingReviews.alignment;
    if (!review) return;
    
    const segment = review.payload.segments?.[index];
    if (!segment || isNaN(value)) return;
    
    segment[field] = value;
    if (field === "start" && typeof segment.end === "number" && segment.end < value) {
      segment.end = value;
    }
    if (field === "end" && typeof segment.start === "number" && segment.start > value) {
      segment.start = value;
    }
    
    if (metaEl) {
      metaEl.textContent = `Segment ${index + 1} (${utils.formatTime(segment.start)} — ${utils.formatTime(segment.end)})`;
    }
  },

  updateSpeaker(index, speaker) {
    const review = state.pendingReviews.alignment;
    if (!review) return;
    
    const segment = review.payload.segments?.[index];
    if (!segment) return;
    
    const normalized = speaker.trim();
    segment.speaker_id = normalized || null;
    this.refreshSpeakers([normalized]);
  },

  updateText(index, text) {
    const review = state.pendingReviews.alignment;
    if (!review) return;
    
    const segment = review.payload.segments?.[index];
    if (!segment) return;
    
    segment.text = text;
  },

  async submit(applyChanges) {
    const review = state.pendingReviews.alignment;
    if (!review?.runId) return false;
    
    const base = applyChanges ? review.payload : review.original;
    const submission = utils.clone(base);
    
    if (applyChanges && Array.isArray(submission.segments)) {
      submission.segments.forEach(seg => { seg.words = null; });
    }
    
    if (Array.isArray(submission.segments)) {
      submission.segments.sort((a, b) => {
        const startA = typeof a.start === "number" ? a.start : 0;
        const startB = typeof b.start === "number" ? b.start : 0;
        return startA - startB;
      });
      
      submission.segments.forEach(seg => {
        if (!seg) return;
        
        if (typeof seg.start !== "number") {
          const parsed = Number(seg.start);
          seg.start = isNaN(parsed) ? 0 : parsed;
        }
        if (typeof seg.end !== "number") {
          const parsed = Number(seg.end);
          seg.end = isNaN(parsed) ? seg.start : parsed;
        }
        if (seg.end < seg.start) seg.end = seg.start;
        if (typeof seg.text !== "string") seg.text = seg.text ? String(seg.text) : "";
        if (!seg.speaker_id?.trim() || seg.speaker_id === undefined) seg.speaker_id = null;
      });
    }
    
    [el.alignment.apply, el.alignment.skip, el.alignment.add].forEach(btn => {
      if (btn) btn.disabled = true;
    });
    if (el.alignment.status) el.alignment.status.textContent = "Submitting alignment review…";
    
    try {
      const response = await fetch(API.routes.reviewAlignment, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: review.runId, alignment: submission })
      });
      
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      
      ui.log(applyChanges ? "🪄 Applied alignment adjustments." : "👍 Alignment accepted without changes.");
      this.hide();
      ui.setStatus("Running", "running");
      return true;
    } catch (err) {
      if (el.alignment.status) {
        el.alignment.status.textContent = `Failed to submit alignment review: ${err.message}`;
      }
      [el.alignment.apply, el.alignment.skip, el.alignment.add].forEach(btn => {
        if (btn) btn.disabled = false;
      });
      ui.log(`❌ Failed to submit alignment review: ${err.message}`);
      return false;
    }
  }
};

// TTS Review
const ttsReview = {
  ...createReviewManager("tts", el.tts),

  show(event) {
    if (!el.tts.card || !el.tts.body) return;
    
    const runId = event?.run_id || "";
    const language = event?.language || "";
    const ttsModel = event?.tts_model || "";
    
    let supportedLangs = Array.isArray(event?.languages)
      ? [...new Set(event.languages.map(l => String(l).trim().toLowerCase()).filter(Boolean))]
      : [];
    
    supportedLangs.forEach(code => lang.register(code));
    supportedLangs.sort((a, b) =>
      (lang.toLabel(a) || a).localeCompare(lang.toLabel(b) || b)
    );
    
    const segments = Array.isArray(event?.segments) ? [...event.segments] : [];
    segments.sort((a, b) => {
      const startA = typeof a.start === "number" ? a.start : 0;
      const startB = typeof b.start === "number" ? b.start : 0;
      return startA - startB;
    });
    
    state.pendingReviews.tts = {
      runId,
      language,
      ttsModel,
      languages: supportedLangs,
      segmentRefs: new Map()
    };
    
    el.tts.body.innerHTML = "";
    
    if (!segments.length) {
      const empty = document.createElement("p");
      empty.className = "review-instructions";
      empty.textContent = "No synthesized segments available for review.";
      el.tts.body.appendChild(empty);
    } else {
      segments.forEach((seg, i) => this.renderSegment(seg, i, supportedLangs));
    }
    
    if (el.tts.status) {
      const label = language ? lang.toLabel(language) || language : "";
      const modelLabel = ttsModel ? ` · Model: ${ttsModel}` : "";
      el.tts.status.textContent = label
        ? `Pipeline paused for TTS review (${label})${modelLabel}.`
        : `Pipeline paused for TTS review${modelLabel}.`;
    }
    
    [el.tts.apply, el.tts.skip].forEach(btn => {
      if (btn) btn.disabled = false;
    });
    
    el.tts.card.hidden = false;
    el.tts.card.scrollIntoView({ behavior: "smooth", block: "center" });
  },

  renderSegment(seg, index, supportedLangs) {
    const segId = seg.segment_id || `seg-${index}`;
    const wrapper = document.createElement("div");
    wrapper.className = "review-segment";
    wrapper.dataset.segmentId = segId;
    
    const header = document.createElement("div");
    header.className = "segment-meta";
    const rangeLabel = `${utils.formatTime(seg.start)} — ${utils.formatTime(seg.end)}`;
    const langLabel = seg.lang ? (lang.toLabel(seg.lang) || seg.lang) : "auto";
    header.textContent = `Segment ${index + 1} (${rangeLabel}) · ${langLabel}`;
    
    const audioContainer = document.createElement("div");
    audioContainer.className = "segment-audio";
    let audioEl = null;
    
    if (seg.audio?.url) {
      audioEl = document.createElement("audio");
      audioEl.controls = true;
      audioEl.preload = "metadata";
      audioEl.src = utils.resolveUrl(seg.audio.url);
      audioContainer.appendChild(audioEl);
    } else {
      const note = document.createElement("p");
      note.className = "review-instructions";
      note.textContent = "Audio preview unavailable.";
      audioContainer.appendChild(note);
    }
    
    const textLabel = document.createElement("label");
    textLabel.textContent = "Text";
    const textarea = document.createElement("textarea");
    textarea.dataset.segmentId = segId;
    textarea.value = seg.text || "";
    textarea.rows = Math.max(2, Math.min(6, Math.ceil((textarea.value.length || 1) / 60)));
    textLabel.appendChild(textarea);
    
    const audioPromptSection = document.createElement("div");
    audioPromptSection.className = "audio-prompt-upload";
    
    const audioPromptLabel = document.createElement("label");
    audioPromptLabel.textContent = "Custom audio prompt (optional)";
    audioPromptLabel.className = "audio-prompt-label";
    
    const audioPromptInput = document.createElement("input");
    audioPromptInput.type = "file";
    audioPromptInput.accept = "audio/*";
    audioPromptInput.dataset.segmentId = segId;
    audioPromptInput.dataset.role = "audio-prompt";
    audioPromptInput.className = "audio-prompt-input";
    
    const audioPromptPreview = document.createElement("div");
    audioPromptPreview.className = "audio-prompt-preview";
    audioPromptPreview.dataset.segmentId = segId;
    
    audioPromptInput.onchange = (e) => {
      const file = e.target.files?.[0];
      if (file) {
        const blobUrl = URL.createObjectURL(file);
        audioPromptPreview.innerHTML = "";
        const previewAudio = document.createElement("audio");
        previewAudio.controls = true;
        previewAudio.src = blobUrl;
        previewAudio.className = "audio-prompt-player";
        audioPromptPreview.appendChild(previewAudio);
        
        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "ghost small";
        removeBtn.textContent = "Remove";
        removeBtn.onclick = () => {
          audioPromptInput.value = "";
          audioPromptPreview.innerHTML = "";
          URL.revokeObjectURL(blobUrl);
        };
        audioPromptPreview.appendChild(removeBtn);
      }
    };
    
    audioPromptLabel.appendChild(audioPromptInput);
    audioPromptSection.append(audioPromptLabel, audioPromptPreview);
    
    const controls = document.createElement("div");
    controls.className = "segment-controls";
    
    const langLabelEl = document.createElement("label");
    langLabelEl.textContent = "Language";
    let langControl;
    
    if (supportedLangs.length) {
      langControl = document.createElement("select");
      langControl.dataset.segmentId = segId;
      langControl.dataset.role = "segment-lang";
      langControl.appendChild(new Option("Auto", ""));
      supportedLangs.forEach(code => {
        langControl.appendChild(new Option(lang.toLabel(code) || code, code));
      });
      const normalizedLang = (seg.lang || "").toLowerCase();
      langControl.value = normalizedLang && supportedLangs.includes(normalizedLang) ? normalizedLang : "";
    } else {
      langControl = document.createElement("input");
      langControl.type = "text";
      langControl.dataset.segmentId = segId;
      langControl.dataset.role = "segment-lang";
      langControl.placeholder = "auto";
      langControl.value = seg.lang || "";
      langControl.setAttribute("list", "language-list");
    }
    langLabelEl.appendChild(langControl);
    
    const regenBtn = document.createElement("button");
    regenBtn.type = "button";
    regenBtn.className = "secondary";
    regenBtn.textContent = "Regenerate audio";
    regenBtn.dataset.segmentId = segId;
    regenBtn.onclick = () => this.regenerate(segId);
    
    const statusSpan = document.createElement("span");
    statusSpan.className = "segment-status";
    statusSpan.textContent = "Ready.";
    
    controls.append(langLabelEl, regenBtn, statusSpan);
    wrapper.append(header, audioContainer, textLabel, audioPromptSection, controls);
    el.tts.body.appendChild(wrapper);
    
    state.pendingReviews.tts.segmentRefs.set(segId, {
      wrapper,
      audioEl,
      textArea: textarea,
      langControl,
      regenerateBtn: regenBtn,
      statusLabel: statusSpan,
      audioPromptInput,
      audioPromptPreview
    });
  },

  updateSegment(seg) {
    const review = state.pendingReviews.tts;
    if (!review || !seg?.segment_id) return;
    
    const refs = review.segmentRefs?.get(seg.segment_id);
    if (!refs) return;
    
    if (typeof seg.text === "string" && refs.textArea) {
      refs.textArea.value = seg.text;
    }
    
    if (refs.langControl) {
      const nextLang = (seg.lang || "").toLowerCase();
      if (refs.langControl.tagName === "SELECT") {
        let options = Array.from(refs.langControl.options).map(opt => opt.value);
        if (nextLang && !options.includes(nextLang)) {
          refs.langControl.appendChild(new Option(lang.toLabel(nextLang) || nextLang, nextLang));
          options = Array.from(refs.langControl.options).map(opt => opt.value);
        }
        refs.langControl.value = nextLang && options.includes(nextLang) ? nextLang : "";
      } else {
        refs.langControl.value = seg.lang || "";
      }
    }
    
    if (refs.audioEl && seg.audio?.url) {
      const nextSrc = utils.resolveUrl(seg.audio.url);
      if (nextSrc) {
        refs.audioEl.src = nextSrc;
        refs.audioEl.load();
      }
    }
    
    if (refs.statusLabel) refs.statusLabel.textContent = "Audio regenerated.";
  },

  async regenerate(segmentId) {
    const review = state.pendingReviews.tts;
    if (!review || !segmentId) return;
    
    const refs = review.segmentRefs?.get(segmentId);
    if (!refs) return;
    
    const textValue = refs.textArea?.value.trim() || "";
    const langRaw = refs.langControl?.value.toString().trim() || "";
    const langValue = lang.resolve(langRaw) || langRaw;
    
    if (refs.regenerateBtn) refs.regenerateBtn.disabled = true;
    if (refs.statusLabel) refs.statusLabel.textContent = "Regenerating…";
    
    try {
      // Build FormData to support audio file upload
      const formData = new FormData();
      formData.append("run_id", review.runId);
      formData.append("segment_id", segmentId);
      formData.append("text", textValue);
      
      if (review.language) {
        formData.append("language", review.language);
      }
      
      if (langValue) {
        formData.append("lang", langValue);
      }
      
      // Add custom audio prompt file if uploaded
      if (refs.audioPromptInput?.files?.length > 0) {
        formData.append("audio_prompt_file", refs.audioPromptInput.files[0]);
      }
      
      const response = await fetch(API.routes.regenerateTTS, {
        method: "POST",
        body: formData
      });
      
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      
      const data = await response.json();
      if (data.segment) this.updateSegment(data.segment);
      ui.log(`🔄 Regenerated TTS segment ${segmentId}.`);
    } catch (err) {
      if (refs.statusLabel) refs.statusLabel.textContent = `Failed: ${err.message}`;
      ui.log(`❌ Failed to regenerate TTS segment ${segmentId}: ${err.message}`);
    } finally {
      if (refs.regenerateBtn) refs.regenerateBtn.disabled = false;
    }
  },

  async submit(applyChanges) {
    const review = state.pendingReviews.tts;
    if (!review?.runId) return false;
    
    const segments = [];
    if (el.tts.body) {
      el.tts.body.querySelectorAll(".review-segment").forEach(segEl => {
        const segId = segEl.dataset.segmentId;
        if (!segId) return;
        
        const textArea = segEl.querySelector(`textarea[data-segment-id="${segId}"]`);
        const langControl = segEl.querySelector(`[data-role="segment-lang"]`);
        const textValue = textArea?.value.trim() || "";
        const langInputRaw = langControl?.value.toString().trim() || "";
        const resolvedLang = lang.resolve(langInputRaw) || langInputRaw;
        
        segments.push({
          segment_id: segId,
          text: textValue,
          lang: resolvedLang || null
        });
      });
    }
    
    [el.tts.apply, el.tts.skip].forEach(btn => {
      if (btn) btn.disabled = true;
    });
    if (el.tts.status) el.tts.status.textContent = "Submitting TTS review…";
    
    try {
      const response = await fetch(API.routes.reviewTTS, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: review.runId,
          language: review.language || null,
          segments
        })
      });
      
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      
      ui.log(applyChanges ? "🎧 Applied TTS adjustments." : "👍 Approved TTS segments without changes.");
      this.hide();
      ui.setStatus("Running", "running");
      return true;
    } catch (err) {
      if (el.tts.status) el.tts.status.textContent = `Failed to submit TTS review: ${err.message}`;
      [el.tts.apply, el.tts.skip].forEach(btn => {
        if (btn) btn.disabled = false;
      });
      ui.log(`❌ Failed to submit TTS review: ${err.message}`);
      return false;
    }
  }
};

export { createReviewManager, transcriptionReview, alignmentReview, ttsReview };
