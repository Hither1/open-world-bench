/* Task-specific recordings, with chapter seeks and lossless multipart playback. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const details = $("task-recording");
  const video = $("reference-video");
  let task = null, recording = null, partIndex = 0, selectedMilestone = null, onChapter = null;

  function time(seconds) {
    const value = Math.max(0, Math.floor(seconds));
    return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
  }

  function option(value, label) {
    const el = document.createElement("option");
    el.value = String(value);
    el.textContent = label;
    return el;
  }

  function unload() {
    video.onloadedmetadata = null;
    video.pause();
    video.removeAttribute("src");
    video.load();
  }

  function open() {
    details.open = true;
    details.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
  }

  function progress() {
    if (!recording) return;
    const position = recording.parts[partIndex].start + (video.currentTime || 0);
    $("reference-progress").textContent = `${time(position)} / ${time(recording.duration)}${recording.parts.length > 1 ? ` · Part ${partIndex + 1} of ${recording.parts.length}` : ""}`;
  }

  function play() {
    video.play().catch((error) => {
      if (error.name !== "AbortError") $("reference-message").textContent = "Press Play to continue this recording.";
    });
  }

  function loadPart(index, seek = 0, autoplay = false) {
    if (!recording) return;
    const part = recording.parts[index];
    partIndex = index;
    $("reference-part").value = String(index);
    $("reference-download").href = part.src;
    $("reference-download").textContent = recording.parts.length > 1 ? "Open this part as MP4" : "Open MP4";
    $("reference-message").textContent = "";
    const expectedURL = new URL(part.src, document.baseURI).href;
    const position = Math.min(Math.max(0, seek), part.end - part.start - .05);
    if (video.currentSrc === expectedURL && video.readyState >= 1) {
      video.currentTime = position;
      if (autoplay) play();
      progress();
      return;
    }
    video.pause();
    video.onloadedmetadata = () => {
      if (video.currentSrc !== expectedURL) return;
      video.currentTime = Math.min(position, Math.max(0, video.duration - .05));
      progress();
      if (autoplay) play();
    };
    video.src = part.src;
    video.load();
    progress();
  }

  function seekChapter(id) {
    const chapter = recording?.chapters.find((c) => c.id === id);
    if (!chapter) return;
    open();
    const index = recording.parts.findIndex((part) => chapter.time >= part.start && chapter.time < part.end);
    if (index < 0) return;
    loadPart(index, chapter.time - recording.parts[index].start, true);
    $("reference-chapter").value = id;
    if (onChapter) onChapter(id);
    video.focus({ preventScroll: true });
  }

  function markMilestone(id) {
    selectedMilestone = id;
    const chapter = recording?.chapters.find((c) => c.id === id);
    $("watch-milestone").hidden = !chapter;
    if (chapter) $("watch-milestone").textContent = `Watch ${id} at ${time(chapter.time)}`;
  }

  function chooseRecording(index) {
    unload();
    recording = task.recordings[index];
    partIndex = 0;
    $("reference-choice").value = String(index);
    $("recording-summary-meta").textContent = `${recording.label} · ${time(recording.duration)}`;
    $("reference-credit").textContent = recording.credit;
    $("reference-renderer").textContent = recording.renderer;
    $("reference-scope").textContent = recording.scope;
    $("reference-provenance").href = recording.provenance;
    $("reference-message").textContent = "";
    video.poster = recording.poster;
    video.setAttribute("aria-label", `${task.title}: ${recording.label}`);
    $("reference-part-label").hidden = recording.parts.length === 1;
    $("reference-part").replaceChildren(...recording.parts.map((part, i) => option(i, `Part ${i + 1} of ${recording.parts.length} · ${time(part.start)}–${time(part.end)}`)));
    $("reference-chapter-label").hidden = recording.chapters.length === 0;
    $("reference-chapter").replaceChildren(option("", "Jump to a recorded milestone"), ...recording.chapters.map((chapter) => option(chapter.id, `${time(chapter.time)} · ${chapter.id} · ${chapter.title}`)));
    $("reference-download").href = recording.parts[0].src;
    $("reference-download").textContent = recording.parts.length > 1 ? "Open this part as MP4" : "Open MP4";
    $("reference-part-note").hidden = recording.parts.length === 1;
    markMilestone(selectedMilestone);
    progress();
    if (details.open) loadPart(0);
  }

  function show(nextTask, chapterCallback) {
    unload();
    task = nextTask;
    onChapter = chapterCallback;
    recording = null;
    selectedMilestone = null;
    const available = task.recordings.length > 0;
    $("reference-player-body").hidden = !available;
    $("reference-unavailable").hidden = available;
    $("show-task-recording").textContent = available ? "Watch task recording" : "Recording status";
    $("recording-summary-title").textContent = task.id === "ender-dragon" ? "Model attempt video" : task.id === "lemon-tart" ? "Procedure video" : "Reference video";
    if (!available) {
      $("recording-summary-meta").textContent = "No reference video yet";
      $("reference-unavailable-note").textContent = task.recordingUnavailable.reason;
      $("reference-unavailable-source").href = task.recordingUnavailable.source;
      markMilestone(null);
      return;
    }
    $("reference-choice-label").hidden = task.recordings.length === 1;
    $("reference-choice").replaceChildren(...task.recordings.map((entry, index) => option(index, entry.label)));
    chooseRecording(0);
  }

  details.addEventListener("toggle", () => {
    if (!details.open) video.pause();
    else if (recording && !video.getAttribute("src")) loadPart(partIndex);
  });
  $("show-task-recording").addEventListener("click", open);
  $("watch-milestone").addEventListener("click", () => seekChapter(selectedMilestone));
  $("reference-choice").addEventListener("change", (event) => chooseRecording(Number(event.target.value)));
  $("reference-part").addEventListener("change", (event) => {
    $("reference-chapter").value = "";
    loadPart(Number(event.target.value), 0, !video.paused);
  });
  $("reference-chapter").addEventListener("change", (event) => seekChapter(event.target.value));
  video.addEventListener("timeupdate", progress);
  video.addEventListener("ended", () => {
    if (recording && partIndex + 1 < recording.parts.length) loadPart(partIndex + 1, 0, true);
  });
  video.addEventListener("error", () => {
    if (recording && video.getAttribute("src")) $("reference-message").textContent = "This recording could not load. Try opening the MP4 directly using the link below.";
  });
  window.OWBRecordings = { show, markMilestone, clear() { recording = null; task = null; unload(); } };
})();
