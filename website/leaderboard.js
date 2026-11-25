(() => {
  "use strict";
  const data = window.OWB_LEADERBOARD;
  const root = document.getElementById("leaderboard-interactive");
  if (!data || !root) return;

  let condition = data.conditions[0].id;
  let world = "overall";
  const expanded = new Set();
  const overall = { id: "overall", name: "Overall", short: "Σ" };
  const worlds = [overall, ...data.worlds];
  const byId = (id) => document.getElementById(id);
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // Zero is a valid published score; missing or malformed data is not.
  function resultFor(model, worldId) {
    const result = model.results[condition]?.[worldId];
    if (!result || !Number.isInteger(result.successes) || !Number.isInteger(result.episodes)
      || result.episodes <= 0 || result.successes < 0 || result.successes > result.episodes) return null;
    return { ...result, rate: 100 * result.successes / result.episodes };
  }

  const percentage = (result) => `${Number(result.rate.toFixed(1))}%`;

  function score(result) {
    const node = element("span", "lb-score");
    node.append(element("strong", "", result ? percentage(result) : "—"));
    node.append(element("small", "", result ? `${result.successes} / ${result.episodes}` : "Pending"));
    node.setAttribute("aria-label", result
      ? `${percentage(result)}; ${result.successes} successful out of ${result.episodes} evaluated episodes`
      : "Pending or unscored");
    return node;
  }

  function reportLink(result) {
    if (!result?.reportUrl) return null;
    try {
      const url = new URL(result.reportUrl, document.baseURI);
      if (!["http:", "https:"].includes(url.protocol)) return null;
      const link = element("a", "lb-report", "Evaluation report");
      link.href = url.href;
      return link;
    } catch { return null; }
  }

  function modelDetails(model) {
    const panel = element("div", "lb-detail");
    panel.append(element("p", "lb-detail-title", `${model.name} · ${data.conditions.find((item) => item.id === condition).name}`));
    const grid = element("div", "lb-detail-grid");
    for (const item of worlds) {
      const card = element("div", "lb-detail-card");
      const result = resultFor(model, item.id);
      card.append(element("span", "lb-detail-world", item.name), score(result));
      const link = reportLink(result);
      if (link) card.append(link);
      grid.append(card);
    }
    panel.append(grid);
    return panel;
  }

  function render() {
    const currentCondition = data.conditions.find((item) => item.id === condition);
    const currentWorld = worlds.find((item) => item.id === world);
    const models = data.models.map((model) => ({ model, result: resultFor(model, world) }))
      .sort((a, b) => (b.result?.rate ?? -1) - (a.result?.rate ?? -1));
    const published = models.filter((item) => item.result).length;
    byId("lb-chart-title").textContent = world === "overall" ? "Overall task success" : `${currentWorld.name} task success`;
    byId("lb-chart-subtitle").textContent = world === "overall"
      ? "Across all five worlds · published aggregates only"
      : `${currentCondition.name} · successful / evaluated episodes`;
    byId("lb-condition-note").textContent = currentCondition.description;
    byId("lb-status").textContent = published ? `${published} / ${models.length} models scored` : "Results pending";
    byId("lb-table-condition").textContent = currentCondition.name;
    byId("lb-table-caption").textContent = "Task success rate (%) · successful / evaluated episodes";
    byId("lb-conditions").querySelectorAll("button").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.condition === condition));
    });

    const rows = document.createDocumentFragment();
    let rank = 0;
    let previousRate = null;
    models.forEach(({ model, result }, index) => {
      if (result && result.rate !== previousRate) rank = index + 1;
      previousRate = result?.rate ?? null;
      const row = element("details", `lb-model lb-tone-${model.tone}`);
      row.dataset.model = model.id;
      row.open = expanded.has(model.id);
      row.addEventListener("toggle", () => {
        if (!row.isConnected) return;
        if (row.open) expanded.add(model.id);
        else expanded.delete(model.id);
      });
      const summary = element("summary", "lb-row");
      const ranking = element("span", "lb-rank", result ? String(rank) : "—");
      ranking.setAttribute("aria-label", result ? `Rank ${rank}` : "Unranked");
      const mark = element("span", "lb-mark", model.mark);
      mark.setAttribute("aria-hidden", "true");
      const name = element("span", "lb-name");
      name.append(element("strong", "", model.name), element("small", "", model.organization));
      const track = element("span", `lb-track${result ? "" : " is-pending"}`);
      track.setAttribute("aria-hidden", "true");
      if (result) {
        const fill = element("span", "lb-fill");
        fill.style.width = `${result.rate}%`;
        track.append(fill);
      }
      const chevron = element("span", "lb-chevron");
      chevron.setAttribute("aria-hidden", "true");
      summary.append(ranking, mark, name, track, score(result), chevron);
      row.append(summary, modelDetails(model));
      rows.append(row);
    });
    byId("lb-rows").replaceChildren(rows);

    const header = element("tr");
    const worldHeader = element("th", "", "World");
    worldHeader.scope = "col";
    header.append(worldHeader);
    for (const { model } of models) {
      const cell = element("th", `lb-tone-${model.tone}`);
      cell.scope = "col";
      cell.append(element("span", "lb-column-name", model.name), element("small", "", model.organization));
      header.append(cell);
    }
    byId("lb-table-head").replaceChildren(header);
    const body = document.createDocumentFragment();
    const foot = document.createDocumentFragment();
    for (const item of [...data.worlds, overall]) {
      const row = element("tr", item.id === world ? "is-selected" : "");
      const heading = element("th");
      heading.scope = "row";
      const badge = element("span", "lb-world-code", item.short);
      badge.setAttribute("aria-hidden", "true");
      heading.append(badge, document.createTextNode(item.name));
      row.append(heading);
      for (const { model } of models) {
        const result = resultFor(model, item.id);
        const cell = element("td", result ? "is-scored" : "is-pending");
        cell.append(score(result));
        const link = reportLink(result);
        if (link) cell.append(link);
        row.append(cell);
      }
      (item.id === "overall" ? foot : body).append(row);
    }
    byId("lb-table-body").replaceChildren(body);
    byId("lb-table-foot").replaceChildren(foot);
  }

  for (const item of data.conditions) {
    const button = element("button", "", item.name);
    button.type = "button";
    button.dataset.condition = item.id;
    button.setAttribute("aria-controls", "lb-rows lb-table-body lb-table-foot");
    button.addEventListener("click", () => { condition = item.id; render(); });
    byId("lb-conditions").append(button);
  }
  for (const item of worlds) {
    const option = element("option", "", item.id === "overall" ? "Overall · all worlds" : item.name);
    option.value = item.id;
    byId("lb-world").append(option);
  }
  byId("lb-world").addEventListener("change", (event) => { world = event.target.value; render(); });
  render();
  root.hidden = false;
})();
