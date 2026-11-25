/* Public design diagrams only. This module does not import or run a simulator. */
(() => {
  "use strict";
  const tasks = window.OWB_TASKS;
  if (!Array.isArray(tasks) || !tasks.length) return;
  const worlds = [
    ["teyvat", "Teyvat"], ["wildlands", "Wildlands"], ["skylands", "Skylands"],
    ["minecraft", "Minecraft"], ["mobile-manipulation", "Mobile manipulation"]
  ];
  const $ = (id) => document.getElementById(id);
  let currentTask = null;
  let stageIndex = 0;
  let currentMilestone = null;
  let lastWorld = null;
  const nodeWidth = 174, nodeHeight = 104, stepX = 210, stepY = 148, padding = 26;
  const svgNS = "http://www.w3.org/2000/svg";

  // Native links still open the full designs when JavaScript is unavailable.
  document.querySelectorAll("[data-environment]").forEach((card) => {
    card.href = `#explore/${card.dataset.environment}`;
  });

  function make(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function svg(tag, attrs) {
    const el = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    return el;
  }

  function routeFor(task, stage, milestone) {
    return `#explore/${task.world}/${task.id}${stage ? `/${stage}` : ""}${milestone ? `/${encodeURIComponent(milestone)}` : ""}`;
  }

  function scrollToExplorer(focus) {
    if (focus) focus.focus({ preventScroll: true });
    $("worlds").scrollIntoView({ behavior: "instant", block: "start" });
  }

  function updateScrollHint() {
    const viewport = $("task-map-scroll");
    document.querySelector(".map-scroll-hint").hidden = viewport.scrollWidth <= viewport.clientWidth;
  }

  if ("ResizeObserver" in window) new ResizeObserver(updateScrollHint).observe($("task-map-scroll"));
  else window.addEventListener("resize", updateScrollHint);

  worlds.forEach(([id, name]) => {
    const option = make("option", "", name);
    option.value = id;
    $("explorer-world").append(option);
  });
  $("explorer-world").addEventListener("change", (event) => {
    window.location.hash = `explore/${event.target.value}`;
  });

  function renderTask(task) {
    currentTask = task;
    lastWorld = task.world;
    const worldName = worlds.find(([id]) => id === task.world)[1];
    $("explorer-world").value = task.world;
    $("explorer-world-title").textContent = worldName;
    $("explorer-cover").src = `media/img/${task.world}-cover.webp`;
    $("task-title").textContent = task.title;
    $("task-status").textContent = task.status;
    $("task-goal").textContent = task.goal;
    $("task-metric").textContent = task.metric;
    $("task-source").href = `https://github.com/Hither1/open-world-bench/blob/main/docs/${task.source}`;
    $("task-route-note").textContent = task.route;
    window.OWBRecordings.show(task, (id) => {
      const index = task.phases.findIndex((phase) => phase.ids.includes(id));
      if (index >= 0) selectStage(index, id, true);
    });
    $("task-choices").replaceChildren(...tasks.filter((t) => t.world === task.world).map((t) => {
      const link = make("a", "task-choice", t.title);
      link.href = routeFor(t);
      if (t.id === task.id) link.setAttribute("aria-current", "true");
      return link;
    }));

    const cols = Math.max(...task.phases.map((p) => p.position[0])) + 1;
    const rows = Math.max(...task.phases.map((p) => p.position[1])) + 1;
    const width = cols * stepX - (stepX - nodeWidth) + 2 * padding;
    const height = rows * stepY - (stepY - nodeHeight) + 2 * padding + 28;
    $("task-map").style.width = `${width}px`;
    $("task-map").style.height = `${height}px`;
    $("task-connections").setAttribute("viewBox", `0 0 ${width} ${height}`);
    $("task-map-scroll").scrollLeft = 0;
    $("task-nodes").replaceChildren(...task.phases.map((p, i) => {
      const button = make("button", "task-node");
      button.type = "button";
      button.id = `task-node-${i}`;
      button.style.left = `${padding + p.position[0] * stepX}px`;
      button.style.top = `${padding + 14 + p.position[1] * stepY}px`;
      button.setAttribute("aria-controls", "stage-title stage-milestones milestone-description");
      button.append(make("span", "task-node-title", p.title), make("span", "task-node-count", `${p.ids.length} ${task.metric.includes("milestones") ? (p.ids.length === 1 ? "milestone" : "milestones") : (p.ids.length === 1 ? "outcome" : "outcomes")}`));
      button.addEventListener("click", (event) => {
        selectStage(i, null, true);
        if (event.detail > 0) {
          document.querySelector(".stage-inspector").scrollIntoView({
            block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth"
          });
        }
      });
      button.addEventListener("keydown", (event) => {
        let next;
        if (["ArrowRight", "ArrowDown"].includes(event.key)) next = Math.min(task.phases.length - 1, i + 1);
        if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = Math.max(0, i - 1);
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = task.phases.length - 1;
        if (next === undefined) return;
        event.preventDefault();
        selectStage(next, null, true);
        $(`task-node-${next}`).focus({ preventScroll: true });
        revealNode(next);
      });
      return button;
    }));
  }

  function drawConnections() {
    const root = $("task-connections");
    root.replaceChildren();
    const defs = svg("defs", {});
    for (const kind of ["route", "memory"]) {
      const marker = svg("marker", { id: `task-arrow-${kind}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" });
      marker.append(svg("path", { d: "M 0 0 L 10 5 L 0 10 z", class: `arrow-${kind}` }));
      defs.append(marker);
    }
    root.append(defs);
    const center = (index) => {
      const p = currentTask.phases[index].position;
      return { x: padding + p[0] * stepX, y: padding + 14 + p[1] * stepY };
    };
    function path(from, to, memory = false) {
      const a = center(from), b = center(to);
      let d;
      if (memory) {
        if (a.x === b.x) {
          const x = a.x - 22;
          d = `M ${a.x} ${a.y + 34} C ${x} ${a.y + 34}, ${x} ${b.y + 34}, ${b.x} ${b.y + 34}`;
        } else {
          // Route delayed-use links through the top margin and column gaps,
          // keeping them visible rather than drawing through intervening nodes.
          const forward = b.x > a.x;
          const x1 = a.x + (forward ? nodeWidth : 0), x2 = b.x + (forward ? 0 : nodeWidth);
          const c1 = x1 + (forward ? 14 : -14), c2 = x2 + (forward ? -14 : 14);
          d = `M ${x1} ${a.y + 30} H ${c1} V 16 H ${c2} V ${b.y + 30} H ${x2}`;
        }
      } else if (a.x === b.x) {
        const down = b.y > a.y;
        d = `M ${a.x + nodeWidth / 2} ${a.y + (down ? nodeHeight : 0)} L ${b.x + nodeWidth / 2} ${b.y + (down ? 0 : nodeHeight)}`;
      } else {
        const forward = b.x > a.x;
        const x1 = a.x + (forward ? nodeWidth : 0), x2 = b.x + (forward ? 0 : nodeWidth);
        const mid = (x1 + x2) / 2;
        d = `M ${x1} ${a.y + nodeHeight / 2} C ${mid} ${a.y + nodeHeight / 2}, ${mid} ${b.y + nodeHeight / 2}, ${x2} ${b.y + nodeHeight / 2}`;
      }
      const active = from === stageIndex || to === stageIndex;
      root.append(svg("path", { d, class: `task-edge${memory ? " memory-edge" : ""}${active ? " connected" : ""}`, "marker-end": `url(#task-arrow-${memory ? "memory" : "route"})` }));
    }
    currentTask.edges.forEach(([from, to]) => path(from, to));
    if ($("memory-links").checked) currentTask.memory.forEach(([from, to]) => path(from, to, true));
  }

  function selectMilestone(id, updateURL) {
    const milestone = currentTask.milestones.find((m) => m.id === id);
    currentMilestone = id;
    window.OWBRecordings.markMilestone(id);
    $("milestone-id").textContent = milestone.id;
    $("milestone-title").textContent = milestone.title;
    $("milestone-description").textContent = milestone.detail;
    $("stage-milestones").querySelectorAll("button").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.milestone === id));
    });
    if (updateURL) replaceRoute();
  }

  function replaceRoute() {
    history.replaceState(null, "", routeFor(currentTask, currentTask.phases[stageIndex].id, currentMilestone));
  }

  function selectStage(index, milestoneId, updateURL) {
    stageIndex = index;
    const stage = currentTask.phases[index];
    $("task-nodes").querySelectorAll("button").forEach((node, i) => {
      node.setAttribute("aria-pressed", String(i === index));
      const related = currentTask.edges.some(([from, to]) => (from === index && to === i) || (to === index && from === i));
      node.classList.toggle("related", related);
    });
    $("stage-position").textContent = `Stage ${index + 1} of ${currentTask.phases.length}`;
    $("stage-title").textContent = stage.title;
    $("stage-note").textContent = stage.note;
    $("stage-previous").disabled = index === 0;
    $("stage-next").disabled = index === currentTask.phases.length - 1;
    $("stage-milestones").replaceChildren(...stage.ids.map((id) => {
      const milestone = currentTask.milestones.find((m) => m.id === id);
      const button = make("button", "milestone-choice");
      button.type = "button";
      button.dataset.milestone = id;
      button.setAttribute("aria-controls", "milestone-description");
      button.append(make("span", "milestone-code", id), make("span", "", milestone.title));
      button.addEventListener("click", () => selectMilestone(id, true));
      return button;
    }));
    const memories = currentTask.memory.filter(([from, to]) => from === index || to === index);
    const memoryPanel = $("stage-memory");
    memoryPanel.replaceChildren();
    if (memories.length) {
      memoryPanel.append(make("h4", "", "What carries forward"));
      memories.forEach(([from, to, description]) => {
        const row = make("div", "memory-connection");
        row.append(make("p", "", description));
        const target = from === index ? to : from;
        const link = make("button", "memory-jump", `${from === index ? "Used at" : "Learned at"}: ${currentTask.phases[target].title}`);
        link.type = "button";
        link.addEventListener("click", () => {
          selectStage(target, null, true);
          $(`task-node-${target}`).focus({ preventScroll: true });
          revealNode(target);
        });
        row.append(link);
        memoryPanel.append(row);
      });
    }
    memoryPanel.hidden = memories.length === 0;
    selectMilestone(stage.ids.includes(milestoneId) ? milestoneId : stage.ids[0], updateURL);
    drawConnections();
  }

  function revealNode(index) {
    const node = $(`task-node-${index}`);
    const viewport = $("task-map-scroll");
    const left = node.offsetLeft, right = left + nodeWidth;
    if (left < viewport.scrollLeft) viewport.scrollLeft = Math.max(0, left - padding);
    else if (right > viewport.scrollLeft + viewport.clientWidth) viewport.scrollLeft = right - viewport.clientWidth + padding;
  }

  $("stage-previous").addEventListener("click", () => {
    if (stageIndex > 0) { selectStage(stageIndex - 1, null, true); revealNode(stageIndex); }
  });
  $("stage-next").addEventListener("click", () => {
    if (stageIndex < currentTask.phases.length - 1) { selectStage(stageIndex + 1, null, true); revealNode(stageIndex); }
  });
  $("memory-links").addEventListener("change", () => {
    $("memory-legend").hidden = !$("memory-links").checked;
    if (currentTask) drawConnections();
  });

  function route(initial = false) {
    const parts = window.location.hash.slice(1).split("/");
    const showTask = parts[0] === "explore" && worlds.some(([id]) => id === parts[1]);
    $("environment-overview").hidden = showTask;
    $("task-explorer-panel").hidden = !showTask;
    if (!showTask) {
      window.OWBRecordings.clear();
      currentTask = null;
      if (parts[0] === "explore") history.replaceState(null, "", "#worlds");
      if (!initial && (parts[0] === "worlds" || parts[0] === "explore")) {
        const previousCard = document.querySelector(`[data-environment="${lastWorld}"]`);
        scrollToExplorer(previousCard || $("explorer-heading"));
      }
      return;
    }
    const options = tasks.filter((t) => t.world === parts[1]);
    const task = options.find((t) => t.id === parts[2]) || options[0];
    const changed = task !== currentTask;
    if (changed) renderTask(task);
    const index = Math.max(0, task.phases.findIndex((p) => p.id === parts[3]));
    let milestoneId;
    try { milestoneId = decodeURIComponent(parts[4] || ""); } catch { milestoneId = ""; }
    selectStage(index, milestoneId, false);
    revealNode(index);
    updateScrollHint();
    replaceRoute();
    // Preserve focus during initial load; later navigation announces the new task.
    scrollToExplorer(initial ? null : $("task-title"));
  }

  window.addEventListener("hashchange", () => route());
  route(true);
})();
