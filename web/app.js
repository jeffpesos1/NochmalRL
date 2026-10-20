/* Noch mal! RL – web UI (game + training dashboard). Vanilla JS, no build step. */
(function () {
  const $ = s => document.querySelector(s);
  const $$ = s => Array.from(document.querySelectorAll(s));
  const api = async (path, opts) => {
    const r = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (e) { }
      throw new Error(msg);
    }
    return r.json();
  };
  const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });
  let toastTimer;
  function toast(msg, err) {
    const t = $("#toast"); t.textContent = msg; t.className = "toast" + (err ? " error" : ""); t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.hidden = true, err ? 4000 : 2000);
  }
  const COLOR_LABEL = { joker: "Joker" };

  // ======================================================================= state
  const S = {
    boards: [], agents: [], board: null, gameId: null, game: null,
    selected: new Set(), legal: [], legalCells: new Set(), suggestions: null, busy: false,
    diceMode: "auto", autoAgents: true, manualDice: null, autoConfirm: false,
    diceFilter: { color: null, size: null }, manualPick: { colors: [], numbers: [] },
  };
  const isMobile = () => window.matchMedia("(max-width: 900px)").matches;

  // ======================================================================= tabs
  $$(".tab").forEach(b => b.addEventListener("click", () => {
    $$(".tab").forEach(x => x.classList.toggle("active", x === b));
    $("#view-game").hidden = b.dataset.view !== "game";
    $("#view-training").hidden = b.dataset.view !== "training";
    $("#view-arena").hidden = b.dataset.view !== "arena";
    $("#view-eval").hidden = b.dataset.view !== "eval";
    if (b.dataset.view === "training") Training.show();
    if (b.dataset.view === "arena") Arena.show();
    if (b.dataset.view === "eval") Eval.show();
  }));

  // ======================================================================= setup
  async function init() {
    try {
      const info = await api("/api/info");
      const badge = $("#device-badge");
      badge.textContent = info.cuda ? `GPU · ${info.gpu}` : "CPU";
      badge.classList.toggle("gpu", !!info.cuda);
    } catch (e) { }
    S.boards = await api("/api/boards");
    S.agents = await api("/api/agents");
    try { S.champion = (await api("/api/champion")).agent; } catch (e) { S.champion = null; }
    for (const a of S.agents) if (a.id === S.champion) a.label = "🏆 " + a.label;
    const bs = $("#board-select"); bs.innerHTML = "";
    const tb = $("#tf-board"); tb.innerHTML = "";
    for (const b of S.boards) {
      const label = (b.official ? "" : "⚙ ") + b.name;
      bs.appendChild(new Option(label, b.id));
      tb.appendChild(new Option(label, b.id));
    }
    renderPlayerRows([{ name: "Du", type: "human" }, { name: "Agent", type: "agent", agent: defaultAgent() }]);
    // deep links: ?view=training opens the dashboard, ?demo=1 starts a human-vs-agent game immediately
    const q = new URLSearchParams(location.search);
    if (["training", "arena", "eval"].includes(q.get("view"))) $$(".tab").find(t => t.dataset.view === q.get("view")).click();
    if (q.get("dice") === "manual") $("#dice-mode").value = "manual";
    if (q.get("demo") === "extern") renderPlayerRows([{ name: "Agent", type: "agent", agent: defaultAgent() }, { name: "Opa", type: "extern" }, { name: "Du", type: "human" }]);
    if (q.get("demo")) $("#start-game").click();
  }
  function defaultAgent() {
    if (S.champion && S.agents.some(a => a.id === S.champion)) return S.champion;
    const trained = S.agents.filter(a => a.id !== "heuristic" && a.id !== "random");
    // prefer checkpoints trained with the currently selected rule variant, then by eval score
    const wantStrict = !($("#opt-start-always") && $("#opt-start-always").checked) && (parseInt(($("#opt-free-rounds") || {}).value, 10) || 0) === 0;
    const blind = ($("#opt-opp-view") && $("#opt-opp-view").value === "none") || $$(".player-row .ptype").some(t => t.value === "extern");
    const wantView = blind ? "none" : "full";
    trained.sort((a, b) => ((b.strict_rules === wantStrict) - (a.strict_rules === wantStrict)) ||
      (((b.opponent_view || "full") === wantView) - ((a.opponent_view || "full") === wantView)) ||
      ((b.eval_score || -1e9) - (a.eval_score || -1e9)));
    return trained.length ? trained[0].id : "heuristic";
  }
  function playerRowsData() {
    return $$(".player-row").map(r => ({
      name: r.querySelector(".pname").value || "Spieler",
      type: r.querySelector(".ptype").value,
      agent: r.querySelector(".agent-select").value,
    }));
  }
  function renderPlayerRows(players) {
    const list = $("#player-list"); list.innerHTML = "";
    players.forEach((p, i) => {
      const row = document.createElement("div"); row.className = "player-row";
      row.innerHTML = `<input class="pname" value="${p.name}" placeholder="Name">
        <select class="ptype"><option value="human">Mensch</option><option value="agent">Agent</option><option value="extern">Extern</option></select>
        <select class="agent-select"></select>`;
      row.querySelector(".ptype").value = p.type;
      const as = row.querySelector(".agent-select");
      for (const a of S.agents) as.appendChild(new Option(a.label, a.id));
      as.value = p.agent || defaultAgent();
      const sync = () => { as.hidden = row.querySelector(".ptype").value !== "agent"; };
      row.querySelector(".ptype").addEventListener("change", sync); sync();
      list.appendChild(row);
    });
  }
  $("#add-player").addEventListener("click", () => {
    const rows = playerRowsData(); if (rows.length >= 6) return;
    rows.push({ name: `Agent ${rows.length}`, type: "agent", agent: defaultAgent() }); renderPlayerRows(rows);
  });
  $("#remove-player").addEventListener("click", () => {
    const rows = playerRowsData(); if (rows.length <= 1) return; rows.pop(); renderPlayerRows(rows);
  });
  $("#start-game").addEventListener("click", async () => {
    try {
      S.diceMode = $("#dice-mode").value; S.autoAgents = $("#auto-agents").checked; S.autoConfirm = $("#auto-confirm").checked;
      const players = playerRowsData().map(p => ({ name: p.name, type: p.type, agent: p.type === "agent" ? p.agent : defaultAgent() }));
      const boardId = $("#board-select").value;
      S.board = await api(`/api/boards/${boardId}`);
      const res = await post("/api/games", {
        board: boardId, players,
        opponent_view: $("#opt-opp-view").value,
        start_column_always: $("#opt-start-always").checked,
        free_rounds: parseInt($("#opt-free-rounds").value, 10) || 0,
      });
      S.gameId = res.id;
      $("#setup-card").hidden = true;
      ["turn-card", "suggest-card", "history-card"].forEach(id => $("#" + id).hidden = false);
      await act(async () => res.state);
    } catch (e) { toast(e.message, true); }
  });

  // ======================================================================= game flow
  function setGame(g) {
    S.game = g; S.selected.clear(); S.suggestions = null; S.legal = []; S.legalCells.clear();
    S.diceFilter = { color: null, size: null };
    render();
  }
  function legalFiltered() {
    const f = S.diceFilter;
    return S.legal.filter(m => (!f.color || m.color === f.color) && (!f.size || m.size === f.size));
  }
  function recomputeLegalCells() {
    S.legalCells = new Set();
    for (const m of legalFiltered()) for (const c of m.cells) S.legalCells.add(c);
  }
  async function refreshLegal() {
    if (!S.game || S.game.phase !== "move") return;
    const r = await api(`/api/games/${S.gameId}/legal`);
    S.legal = r.moves; recomputeLegalCells();
  }
  function curPlayer() { return S.game && S.game.current_player != null ? S.game.players[S.game.current_player] : null; }
  function humanTurn() { const p = curPlayer(); return p && p.type === "human"; }
  function externTurn() { const p = curPlayer(); return p && p.type === "extern"; }
  function anyHuman() { return S.game.players.some(p => p.type !== "agent"); }

  function autoStep() {
    const g = S.game; if (!g) return null;
    if (g.phase === "dice" && S.diceMode === "auto" && anyHuman()) return () => post(`/api/games/${S.gameId}/roll`);
    if (g.phase === "move" && !humanTurn() && !externTurn() && S.autoAgents && anyHuman()) {
      return async () => (await post(`/api/games/${S.gameId}/agent_play`, { until_human: true, auto_roll: S.diceMode === "auto" })).state;
    }
    return null;
  }
  async function act(fn) {
    if (S.busy) return;
    S.busy = true; render();
    try {
      let st = await fn(); setGame(st);
      let step;
      while ((step = autoStep())) { st = await step(); setGame(st); }
      if (S.game.phase === "move" && humanTurn()) await refreshLegal();
    } catch (e) { toast(e.message, true); }
    finally { S.busy = false; render(); }
  }
  async function submitMove(cells) {
    await act(async () => (await post(`/api/games/${S.gameId}/move`, { cells })).state);
  }
  async function askAgent() {
    try {
      const r = await post(`/api/games/${S.gameId}/suggest`, { top_n: 8 });
      S.suggestions = r; renderSuggestions();
    } catch (e) { toast(e.message, true); }
  }

  // ======================================================================= render
  function colorHex(key) { return key === "joker" ? "#222" : S.board.color_hex[key]; }
  function colorName(key) { return key === "joker" ? "Joker" : S.board.color_names[key]; }

  function render() {
    if (!S.game) return;
    const g = S.game, b = S.board;
    $("#round-label").textContent = `${g.round}${g.max_rounds && g.players.length === 1 ? " / " + g.max_rounds : ""}`;
    const cp = curPlayer();
    $("#turn-label").textContent = g.phase === "finished" ? "Ende" : (g.phase === "dice" ? "Würfeln" : cp.name);
    renderDice();
    renderActions();
    renderBoards();
    renderHistory();
    renderSuggestions();
    renderSelectionInfo();
    renderExternPanel();
  }

  function renderExternPanel() {
    const panel = $("#extern-panel");
    const g = S.game;
    if (!g || g.phase !== "move" || !externTurn()) { panel.hidden = true; panel.innerHTML = ""; return; }
    const b = S.board, p = curPlayer();
    panel.hidden = false; panel.innerHTML = "";
    const h = document.createElement("h3"); h.textContent = `${p.name} (extern): was ist passiert?`; panel.appendChild(h);
    if (g.current_is_active) {
      const t = document.createElement("div"); t.className = "muted"; t.textContent = "Welche 2 Würfel hat der aktive Spieler genommen?"; panel.appendChild(t);
      const grp = document.createElement("div"); grp.className = "grp";
      const cs = document.createElement("select"); cs.id = "ext-color";
      cs.appendChild(new Option("– gepasst –", ""));
      for (const f of g.dice.colors) if (g.pool.colors[f] > 0 && !Array.from(cs.options).some(o => o.value === f)) cs.appendChild(new Option(colorName(f), f));
      const ns = document.createElement("select"); ns.id = "ext-number";
      ns.appendChild(new Option("– gepasst –", ""));
      for (const f of g.dice.numbers) { const k = String(f); if (g.pool.numbers[k] > 0 && !Array.from(ns.options).some(o => o.value === k)) ns.appendChild(new Option(f === "joker" ? "Joker ?" : k, k)); }
      grp.appendChild(cs); grp.appendChild(ns); panel.appendChild(grp);
    }
    const t2 = document.createElement("div"); t2.className = "muted"; t2.textContent = "Angesagte Spalten (vollständig):"; panel.appendChild(t2);
    const cl = document.createElement("div"); cl.className = "claim-list";
    b.columns.forEach((c, i) => {
      const l = document.createElement("label"); if (g.col_claimed[i]) l.classList.add("claimed");
      l.innerHTML = `<input type="checkbox" class="ext-col" value="${i}" ${g.col_claimed[i] ? "disabled" : ""}> ${c}`; cl.appendChild(l);
    });
    panel.appendChild(cl);
    const t3 = document.createElement("div"); t3.className = "muted"; t3.textContent = "Angesagte Farben (vollständig):"; panel.appendChild(t3);
    const fl = document.createElement("div"); fl.className = "claim-list";
    b.color_keys.forEach((k, i) => {
      const l = document.createElement("label"); if (g.color_claimed[i]) l.classList.add("claimed");
      l.innerHTML = `<input type="checkbox" class="ext-color-done" value="${i}" ${g.color_claimed[i] ? "disabled" : ""}> <span class="sw" style="background:${b.color_hex[k]}"></span>${b.color_names[k]}`; fl.appendChild(l);
    });
    panel.appendChild(fl);
    const fin = document.createElement("label"); fin.className = "check";
    fin.innerHTML = `<input type="checkbox" id="ext-finished"> hat die zweite Farbe vervollständigt (Spielende nach dieser Runde)`;
    panel.appendChild(fin);
    panel.appendChild(btn("Übernehmen", "primary wide", submitExtern));
  }
  async function submitExtern() {
    const g = S.game;
    let dice = null;
    if (g.current_is_active) {
      const c = $("#ext-color").value, n = $("#ext-number").value;
      if ((c && !n) || (!c && n)) { toast("Bitte Farb- und Zahlenwürfel angeben (oder beide auf „gepasst“)", true); return; }
      if (c && n) dice = { color: c, number: n === "joker" ? "joker" : parseInt(n, 10) };
    }
    const columns = $$(".ext-col:checked").map(x => +x.value);
    const colors = $$(".ext-color-done:checked").map(x => +x.value);
    const finished = $("#ext-finished").checked;
    await act(async () => (await post(`/api/games/${S.gameId}/extern`, { dice, columns, colors, finished })).state);
  }

  function renderDice() {
    const area = $("#dice-area"); area.innerHTML = "";
    const g = S.game;
    if (g.phase === "dice") {
      if (S.diceMode === "manual") {
        // tap-to-pick chips: one row per die
        const wrap = document.createElement("div"); wrap.className = "chip-rows";
        const cfaces = S.board.color_keys.concat(["joker"]);
        const nfaces = S.board.number_faces.map(String).concat(["joker"]);
        if (S.manualPick.colors.length !== g.n_color_dice) S.manualPick.colors = Array(g.n_color_dice).fill(null);
        if (S.manualPick.numbers.length !== g.n_number_dice) S.manualPick.numbers = Array(g.n_number_dice).fill(null);
        const row = (label, faces, arr, i, kind) => {
          const r = document.createElement("div"); r.className = "chip-row";
          r.innerHTML = `<span class="lbl">${label}</span>`;
          for (const f of faces) {
            const d = document.createElement("div"); d.className = "die pick " + (kind === "c" ? (f === "joker" ? "joker" : "") : (f === "joker" ? "joker" : "number"));
            if (kind === "c") { d.style.background = colorHex(f); d.textContent = f === "joker" ? "✖" : ""; d.title = colorName(f); }
            else d.textContent = f === "joker" ? "?" : f;
            if (arr[i] === f) d.classList.add("sel");
            d.addEventListener("click", () => { arr[i] = f; renderDice(); });
            r.appendChild(d);
          }
          return r;
        };
        for (let i = 0; i < g.n_color_dice; i++) wrap.appendChild(row(`Farbe ${i + 1}`, cfaces, S.manualPick.colors, i, "c"));
        for (let i = 0; i < g.n_number_dice; i++) wrap.appendChild(row(`Zahl ${i + 1}`, nfaces, S.manualPick.numbers, i, "n"));
        area.appendChild(wrap);
        renderActions();
      } else {
        area.innerHTML = `<div class="muted">Noch nicht gewürfelt.</div>`;
      }
      return;
    }
    if (!g.dice) return;
    // which dice are still available: compare faces with pool counts
    const poolC = Object.assign({}, g.pool.colors), poolN = Object.assign({}, g.pool.numbers);
    const grpC = document.createElement("div"); grpC.className = "die-group";
    grpC.innerHTML = `<span class="lbl">Farbe</span>`;
    const filterable = humanTurn();
    for (const f of g.dice.colors) {
      const d = document.createElement("div"); d.className = "die" + (f === "joker" ? " joker" : "");
      d.style.background = colorHex(f); d.textContent = f === "joker" ? "✖" : "";
      d.title = colorName(f);
      if (poolC[f] > 0) poolC[f]--; else d.classList.add("used");
      if (filterable && !d.classList.contains("used") && f !== "joker") {
        d.classList.add("pick");
        if (S.diceFilter.color === f) d.classList.add("filter");
        d.addEventListener("click", () => { S.diceFilter.color = S.diceFilter.color === f ? null : f; S.selected.clear(); recomputeLegalCells(); render(); });
      }
      grpC.appendChild(d);
    }
    const grpN = document.createElement("div"); grpN.className = "die-group";
    grpN.innerHTML = `<span class="lbl">Zahl</span>`;
    for (const f of g.dice.numbers) {
      const key = String(f);
      const d = document.createElement("div"); d.className = "die " + (f === "joker" ? "joker" : "number");
      d.textContent = f === "joker" ? "?" : f;
      if (poolN[key] > 0) poolN[key]--; else d.classList.add("used");
      if (filterable && !d.classList.contains("used") && f !== "joker") {
        d.classList.add("pick");
        if (S.diceFilter.size === f) d.classList.add("filter");
        d.addEventListener("click", () => { S.diceFilter.size = S.diceFilter.size === f ? null : f; S.selected.clear(); recomputeLegalCells(); render(); });
      }
      grpN.appendChild(d);
    }
    area.appendChild(grpC); area.appendChild(grpN);
    if (filterable) area.appendChild(Object.assign(document.createElement("div"), { className: "die-hint", textContent: (S.diceFilter.color || S.diceFilter.size) ? "Filter aktiv – nochmal tippen zum Aufheben." : "Tipp auf Farbe/Zahl filtert die markierbaren Felder." }));
    if (g.free_rounds > 0 && g.round <= g.free_rounds && g.players.length > 1) {
      area.appendChild(Object.assign(document.createElement("div"), { className: "hint", textContent: `Wurf ${g.round} von ${g.free_rounds}: alle dürfen aus allen Würfeln wählen.` }));
    }
  }

  function btn(label, cls, onclick, disabled) {
    const b = document.createElement("button"); b.className = "btn " + (cls || ""); b.textContent = label;
    b.disabled = !!disabled || S.busy; b.addEventListener("click", onclick); return b;
  }
  function renderActions() {
    const a = $("#actions"); a.innerHTML = "";
    const g = S.game;
    if (g.phase === "finished") {
      a.appendChild(btn("Neues Spiel", "primary wide", newGameSetup));
    } else if (g.phase === "dice") {
      if (S.diceMode === "manual") {
        const complete = S.manualPick.colors.every(x => x) && S.manualPick.numbers.every(x => x);
        a.appendChild(btn(complete ? "Würfel übernehmen" : "Alle Würfel antippen …", "primary wide", async () => {
          const colors = S.manualPick.colors.slice();
          const numbers = S.manualPick.numbers.map(v => v === "joker" ? "joker" : parseInt(v, 10));
          S.manualPick = { colors: [], numbers: [] };
          await act(() => post(`/api/games/${S.gameId}/dice`, { colors, numbers }));
        }, !complete));
        a.appendChild(btn("Zufällig würfeln", "ghost wide", () => act(() => post(`/api/games/${S.gameId}/roll`))));
      } else {
        a.appendChild(btn("Würfeln", "primary wide", () => act(() => post(`/api/games/${S.gameId}/roll`))));
      }
    } else if (externTurn()) {
      // the extern panel (below) collects the public information; nothing else to do here
    } else if (humanTurn()) {
      const match = selectedMove();
      a.appendChild(btn(match ? `Zug bestätigen (${match.size} ${colorName(match.color)})` : "Felder wählen …", "primary wide", () => submitMove(Array.from(S.selected)), !match));
      a.appendChild(btn("Passen", "", () => submitMove(null)));
      a.appendChild(btn("Agent fragen", "", askAgent));
      a.appendChild(btn("Agent zieht für mich", "ghost wide", () => act(async () => (await post(`/api/games/${S.gameId}/agent_play`, { until_human: false, auto_roll: false })).state)));
    } else {
      a.appendChild(btn("Agent zieht", "primary wide", () => act(async () => (await post(`/api/games/${S.gameId}/agent_play`, { until_human: false, auto_roll: false })).state)));
      a.appendChild(btn("Zeige Verteilung", "", askAgent));
      a.appendChild(btn("Runde zu Ende", "", () => act(async () => (await post(`/api/games/${S.gameId}/agent_play`, { until_human: true, auto_roll: false })).state)));
      if (!anyHuman()) a.appendChild(btn("Spiel zu Ende simulieren", "ghost wide", () => act(async () => (await post(`/api/games/${S.gameId}/agent_play`, { until_human: true, auto_roll: true })).state)));
    }
    if (g.phase !== "finished") a.appendChild(btn("Spiel abbrechen", "ghost", () => { if (confirm("Spiel wirklich abbrechen?")) newGameSetup(); }));
    a.appendChild(btn("Rückgängig", "ghost", () => act(() => post(`/api/games/${S.gameId}/undo`)), !g.can_undo));
    renderMobileBar();
  }
  function renderMobileBar() {
    const bar = $("#mobile-bar");
    if (!isMobile() || !S.game) { bar.hidden = true; bar.innerHTML = ""; return; }
    bar.hidden = false; bar.innerHTML = "";
    const g = S.game;
    const all = $$("#actions .btn");
    // primary = first action button, then up to three more; the full list stays reachable via "…"
    const primary = all[0] ? all[0].cloneNode(true) : null;
    if (primary) { primary.className = "btn primary"; primary.addEventListener("click", () => all[0].click()); bar.appendChild(primary); }
    const rest = all.slice(1).filter(b => !/abbrechen/i.test(b.textContent));
    for (const b of rest.slice(0, 2)) { const c = b.cloneNode(true); c.className = "btn"; c.addEventListener("click", () => b.click()); bar.appendChild(c); }
    const more = btn("…", "ghost", () => { const a = $("#actions"); a.style.display = a.style.display === "grid" ? "" : "grid"; a.scrollIntoView({ behavior: "smooth", block: "center" }); });
    bar.appendChild(more);
  }
  function newGameSetup() {
    S.game = null; S.gameId = null;
    $("#setup-card").hidden = false;
    ["turn-card", "suggest-card", "history-card"].forEach(id => $("#" + id).hidden = true);
    $("#boards").innerHTML = "";
  }
  function selectedMove() {
    if (!S.selected.size) return null;
    const key = Array.from(S.selected).sort((a, b) => a - b).join(",");
    return S.legal.find(m => m.cells.slice().sort((a, b) => a - b).join(",") === key) || null;
  }
  function renderSelectionInfo() {
    const el = $("#selection-info");
    if (!S.game || S.game.phase !== "move" || !humanTurn()) { el.innerHTML = ""; return; }
    if (!S.selected.size) { el.innerHTML = `${S.legal.length} legale Züge. Klicke auf markierbare Felder.`; return; }
    const m = selectedMove();
    if (m) {
      el.innerHTML = `<b>${m.size} × ${colorName(m.color)}</b>` + (m.jokers ? ` · <b>${m.jokers} Joker</b> nötig` : "") + (m.stars ? ` · ${m.stars} ★` : "");
    } else {
      el.innerHTML = `<span style="color:var(--danger)">Auswahl ist kein gültiger Zug</span> (${S.selected.size} Felder)`;
    }
  }

  function renderBoards() {
    const wrap = $("#boards");
    const g = S.game, b = S.board;
    wrap.innerHTML = "";
    if (g.final) {
      const f = g.final;
      const banner = document.createElement("div"); banner.className = "final-banner";
      const winner = f.ranking.length ? g.players[f.ranking[0]] : null;
      const ext = g.players.filter(p => p.type === "extern").map(p => p.name);
      banner.innerHTML = (winner ? `<div class="big">🏆 ${winner.name}: ${f.scores[f.ranking[0]].toFixed(0)} Punkte${ext.length ? " (beste mitgeführte Wertung)" : ""}</div>` : `<div class="big">Spielende</div>`) +
        `<ol>${f.ranking.map(i => `<li>${g.players[i].name}: <b>${f.scores[i].toFixed(0)}</b> Punkte (${f.jokers_left[i]} Joker übrig)</li>`).join("")}</ol>` +
        (ext.length ? `<div class="muted">Nicht mitgeführt: ${ext.join(", ")} – Punkte bitte vom echten Blatt ablesen.</div>` : "") +
        `<div class="muted">Spielende nach ${f.rounds} Würfen.</div>`;
      wrap.appendChild(banner);
    }
    const order = g.players.map((_, i) => i);
    if (g.current_player != null) order.sort((x, y) => (x === g.current_player ? -1 : y === g.current_player ? 1 : x - y));
    if (isMobile() && g.phase !== "finished" && g.players.length > 1) {
      // phone: full board only for the current seat, the others as a compact score strip (tap to expand)
      const strip = document.createElement("div"); strip.className = "card compact-opp";
      for (const pi of order.slice(1)) {
        const p = g.players[pi];
        const pill = document.createElement("span"); pill.className = "pill" + (g.active_player === pi ? " active" : "");
        pill.textContent = `${p.name}: ${p.score == null ? "?" : p.score.toFixed(0)} Pkt`;
        pill.addEventListener("click", () => { S.expandOpp = S.expandOpp === pi ? null : pi; renderBoards(); });
        strip.appendChild(pill);
      }
      wrap.appendChild(renderBoard(order[0], true));
      wrap.appendChild(strip);
      if (S.expandOpp != null && S.expandOpp !== order[0]) wrap.appendChild(renderBoard(S.expandOpp));
      fitCells();
      return;
    }
    for (const pi of order) wrap.appendChild(renderBoard(pi));
    fitCells();
  }
  function fitCells() {
    // scale cells to the available width on phones (15 columns + gaps must fit without horizontal scrolling)
    const b = S.board; if (!b) return;
    // measure the viewport, not the card: the card may already be blown up by an unfitted grid
    const avail = Math.min(document.documentElement.clientWidth, window.innerWidth) - 48;
    const size = Math.max(12, Math.min(40, Math.floor((avail - (b.width - 1) * 3) / b.width)));
    document.documentElement.style.setProperty("--cell-mobile", size + "px");
  }
  window.addEventListener("resize", () => { if (S.game) { fitCells(); renderMobileBar(); } });
  function renderBoard(pi, big) {
    const g = S.game, b = S.board, p = g.players[pi];
    const isCur = g.current_player === pi || !!big;
    if (p.type === "extern") {
      const card = document.createElement("div");
      card.className = "board-card small" + (isCur ? " current" : "");
      const claims = g.history.filter(e => e.extern && e.player === pi && (e.columns.length || e.colors.length || e.finished));
      const txt = claims.length ? claims.map(e => `R${e.round}: ` + [...e.columns.map(c => "Spalte " + b.columns[c]), ...e.colors.map(c => b.color_names[b.color_keys[c]]), ...(e.finished ? ["2. Farbe voll"] : [])].join(", ")).join(" · ") : "noch keine Ansagen";
      card.innerHTML = `<div class="board-head"><div class="name">${p.name} ${g.active_player === pi && g.phase !== "finished" ? '<span class="pill active">aktiv</span>' : ""}<span class="pill">extern</span></div></div>
        <div class="extern-note">Blatt wird nicht mitgeführt. Ansagen: ${txt}</div>`;
      return card;
    }
    const interactive = isCur && p.type === "human" && g.phase === "move";
    const card = document.createElement("div");
    card.className = "board-card" + (isCur ? " current" : "") + (!isCur && g.players.length > 1 && g.phase !== "finished" ? " small" : "");
    const chips = `<div class="score-chips">
      <div class="chip"><div class="v">${p.col_points}</div><div class="l">Spalten</div></div>
      <div class="chip"><div class="v">${p.color_points}</div><div class="l">Farben</div></div>
      <div class="chip neg"><div class="v">${p.star_penalty}</div><div class="l">Sterne</div></div>
      <div class="chip"><div class="v">+${p.joker_points}</div><div class="l">Joker</div></div>
      <div class="chip total"><div class="v">${p.score == null ? "?" : p.score.toFixed(0)}</div><div class="l">Punkte</div></div></div>`;
    const pills = (g.active_player === pi && g.phase !== "finished" ? `<span class="pill active">aktiv</span>` : "") +
      (p.type === "agent" ? `<span class="pill agent">${p.agent}</span>` : `<span class="pill">Mensch</span>`) +
      (g.opponent_view === "none" && p.type === "agent" ? `<span class="pill">blind</span>` : "");
    card.innerHTML = `<div class="board-head"><div class="name">${p.name} ${pills}</div>${chips}</div>`;
    const gw = document.createElement("div"); gw.className = "grid-wrap";
    const cols = `repeat(${b.width}, var(--cell))`;
    const labels = document.createElement("div"); labels.className = "col-labels"; labels.style.gridTemplateColumns = cols;
    b.columns.forEach((c, i) => { const d = document.createElement("div"); d.textContent = c; if (i === b.start_column) d.classList.add("start"); labels.appendChild(d); });
    const grid = document.createElement("div"); grid.className = "grid"; grid.style.gridTemplateColumns = cols;
    const opp = new Set();
    if (g.players.length > 1 && isCur) {
      g.players.forEach((q, qi) => { if (qi !== pi) q.marked.forEach((m, c) => { if (m) opp.add(c); }); });
    }
    for (let c = 0; c < b.cell_color.length; c++) {
      const cell = document.createElement("div");
      const ck = b.color_keys[b.cell_color[c]];
      cell.className = "cell"; cell.style.background = b.color_hex[ck];
      if (b.cell_star[c]) cell.classList.add("star");
      if (c % b.width === b.start_column) cell.classList.add("start");
      if (p.marked[c]) cell.classList.add("marked");
      if (interactive && S.legalCells.has(c) && !p.marked[c]) cell.classList.add("legal");
      if (interactive && S.selected.has(c)) cell.classList.add("selected");
      cell.dataset.cell = c;
      if (interactive) cell.addEventListener("click", () => {
        if (!S.legalCells.has(c) || p.marked[c]) return;
        if (S.selected.has(c)) S.selected.delete(c); else S.selected.add(c);
        renderActions(); renderSelectionInfo();
        cell.classList.toggle("selected", S.selected.has(c));
        const m = selectedMove();
        if (m && S.autoConfirm) submitMove(Array.from(S.selected));
      });
      grid.appendChild(cell);
    }
    const pts = document.createElement("div"); pts.className = "col-points"; pts.style.gridTemplateColumns = cols;
    for (let i = 0; i < b.width; i++) {
      const d = document.createElement("div");
      const first = document.createElement("span"); first.textContent = b.column_points_first[i];
      const second = document.createElement("span"); second.textContent = b.column_points_second[i];
      if (g.col_claimed[i] && !(p.col_done[i])) first.classList.add("claimed");
      if (p.col_done[i]) first.classList.add("done");
      d.appendChild(first); d.appendChild(second); pts.appendChild(d);
    }
    gw.appendChild(labels); gw.appendChild(grid); gw.appendChild(pts); card.appendChild(gw);
    const foot = document.createElement("div"); foot.className = "board-foot";
    const jok = document.createElement("div"); jok.className = "jokers";
    for (let i = 0; i < b.jokers; i++) { const s = document.createElement("span"); s.textContent = "!"; if (i < p.jokers_used) s.classList.add("used"); jok.appendChild(s); }
    const cd = document.createElement("div"); cd.className = "colors-done";
    b.color_keys.forEach((k, i) => {
      const s = document.createElement("span"); s.style.background = b.color_hex[k];
      const cnt = b.cell_color.reduce((acc, cc, idx) => acc + (cc === i && p.marked[idx] ? 1 : 0), 0);
      const tot = b.cell_color.filter(cc => cc === i).length;
      s.textContent = `${cnt}/${tot}`; s.title = b.color_names[k];
      if (p.color_done[i]) s.classList.add("done"); else if (g.color_claimed[i]) s.classList.add("claimed");
      cd.appendChild(s);
    });
    foot.appendChild(jok); foot.appendChild(cd); card.appendChild(foot);
    if (g.round_log && g.round_log.length && g.phase !== "finished") {
      const e = g.round_log.find(x => x.player === pi);
      if (e) { const d = document.createElement("div"); d.className = "hint"; d.textContent = `Diese Runde: ${describeEntry(e)}`; card.appendChild(d); }
    }
    return card;
  }
  function describeEntry(e) {
    if (e.extern) {
      const b = S.board;
      const parts = [];
      if (e.active) parts.push(e.dice ? `nimmt ${colorName(e.dice.color)} + ${e.dice.number === "joker" ? "?" : e.dice.number}` : "passt");
      parts.push(...e.columns.map(c => "Spalte " + b.columns[c] + " voll"), ...e.colors.map(c => b.color_names[b.color_keys[c]] + " voll"));
      if (e.finished) parts.push("2. Farbe voll → Spielende");
      return parts.length ? parts.join(", ") : "keine Ansage";
    }
    if (e.pass) return "passt";
    return `${e.size} × ${colorName(e.color)}` + (e.jokers ? ` (${e.jokers} Joker)` : "") + (e.round_points ? ` → +${e.round_points} Spalten/Farben` : "");
  }
  function renderHistory() {
    const h = $("#history"); h.innerHTML = "";
    const hist = S.game.history.slice().reverse();
    for (const e of hist) {
      const d = document.createElement("div"); d.className = "h";
      d.innerHTML = `<span>R${e.round} · <b>${e.name}</b>${e.by === "agent" ? " 🤖" : ""}</span><span>${describeEntry(e)}</span>`;
      h.appendChild(d);
    }
    if (!hist.length) h.innerHTML = `<div class="muted">Noch keine Züge.</div>`;
  }
  function renderSuggestions() {
    const box = $("#suggestions"); box.innerHTML = "";
    $("#suggest-agent").textContent = S.suggestions ? `(${S.suggestions.agent})` : "";
    if (!S.suggestions) { box.innerHTML = `<div class="muted">„Agent fragen“ zeigt die Zugverteilung des Agenten für den aktuellen Spieler.</div>`; return; }
    const sug = S.suggestions;
    const items = sug.suggestions.map(s => ({ s, prob: s.prob })).concat([{ pass: true, prob: sug.pass_prob }]).sort((a, b) => b.prob - a.prob);
    for (const it of items) {
      const d = document.createElement("div"); d.className = "sug" + (it.pass ? " pass" : "");
      const desc = it.pass ? "Passen" : `<span class="swatch" style="background:${colorHex(it.s.color)}"></span> ${it.s.size} × ${colorName(it.s.color)}` + (it.s.jokers ? ` · ${it.s.jokers} Joker` : "") + (it.s.stars ? ` · ${it.s.stars}★` : "");
      d.innerHTML = `<div class="desc">${desc}</div><div class="pct">${(it.prob * 100).toFixed(1)} %</div><div class="bar"><i style="width:${(it.prob * 100).toFixed(1)}%"></i></div>`;
      if (!it.pass) {
        d.addEventListener("mouseenter", () => highlight(it.s.cells, true));
        d.addEventListener("mouseleave", () => highlight(it.s.cells, false));
        d.addEventListener("click", () => {
          if (!humanTurn()) return;
          S.selected = new Set(it.s.cells); renderBoards(); renderActions(); renderSelectionInfo();
        });
      }
      box.appendChild(d);
    }
    if (sug.value != null) box.insertAdjacentHTML("beforeend", `<div class="value-line">Erwartete weitere Punkte (Value): <b>${sug.value.toFixed(1)}</b></div>`);
  }
  function highlight(cells, on) {
    const card = $("#boards .board-card.current") || $("#boards .board-card");
    if (!card) return;
    for (const c of cells) { const el = card.querySelector(`.cell[data-cell="${c}"]`); if (el) el.classList.toggle("hint", on); }
  }

  // ======================================================================= training dashboard
  const Training = (function () {
    let runs = [], run = null, records = [], baseline = null, timer = null, range = 0, status = null;
    const compareSel = new Set(); const compareCache = {};
    const accent = "#00d68f", blue = "#4f8cff", red = "#ff4d6d", gray = "#8a94a6", warn = "#ffb547", purple = "#b07cff";
    const PALETTE = [accent, blue, warn, red, purple, "#35d6ff", "#ff8c42", "#a3e635"];

    async function show() {
      await loadRuns();
      await loadAgentsForFinetune();
      if (!timer) timer = setInterval(tick, 3000);
      await tick();
    }
    async function loadAgentsForFinetune() {
      S.agents = await api("/api/agents");
      const sel = $("#ft-base"); const prev = Array.from(sel.selectedOptions).map(o => o.value); sel.innerHTML = "";
      for (const a of S.agents) if (a.id !== "heuristic" && a.id !== "random") sel.appendChild(new Option(a.label, a.id, false, prev.includes(a.id) || (!prev.length && /best\.pt$/.test(a.id))));
      const fb = $("#ft-board"); if (!fb.options.length) for (const b of S.boards) fb.appendChild(new Option(b.name, b.id));
      const eo = $("#tf-eval-opp"); const prevEo = eo.value; eo.innerHTML = "";
      for (const a of S.agents) eo.appendChild(new Option(a.label, a.id)); eo.value = prevEo || "heuristic";
    }
    async function loadRuns() {
      runs = await api("/api/training/runs");
      const sel = $("#run-select"); const prev = sel.value; sel.innerHTML = "";
      for (const r of runs) sel.appendChild(new Option(r.run + (r.running ? " ● läuft" : ""), r.run));
      if (!runs.length) { sel.appendChild(new Option("– noch kein Training –", "")); run = null; return; }
      const running = runs.find(r => r.running);
      run = (prev && runs.some(r => r.run === prev)) ? prev : (running ? running.run : runs[0].run);
      sel.value = run;
      if (!compareSel.size) { compareSel.add(run); runs.slice(0, 4).forEach(r => compareSel.add(r.run)); }
    }
    async function tick() {
      if ($("#view-training").hidden) return;
      try {
        status = await api("/api/training/status" + (run ? "?run=" + encodeURIComponent(run) : ""));
        const pill = $("#train-status");
        pill.textContent = status.runs.length ? `läuft: ${status.runs.join(", ")} (${status.runs.length}/${status.max_parallel})` : "gestoppt";
        pill.classList.toggle("live", status.running);
        const stop = $("#stop-training");
        stop.hidden = !status.running;
        stop.textContent = status.runs.includes(run) ? `Stop ${run}` : "Alle stoppen";
        $("#ft-parallel-info").textContent = `parallel möglich: ${status.max_parallel} Läufe (NOCHMAL_MAX_TRAININGS)`;
        $("#train-console").textContent = status.tail || "–";
        $("#train-console").scrollTop = 1e9;
        renderPipeline(status.pipeline);
        const known = new Set(runs.map(r => r.run));
        if (status.runs.some(r => !known.has(r)) || runs.some(r => r.running !== status.runs.includes(r.run))) await loadRuns();
        if (run) {
          const r = await api(`/api/training/log?run=${encodeURIComponent(run)}`);
          records = r.records.filter(x => x.type === "iter");
          baseline = r.records.find(x => x.type === "baseline") || baseline;
          compareCache[run] = records;
          render();
          await renderCompare();
        } else { $("#kpis").innerHTML = ""; $("#run-header").hidden = true; }
      } catch (e) { console.warn(e); }
    }
    function renderPipeline(p) {
      const el = $("#pipeline-status");
      if (!p || p.stage === "idle") { el.innerHTML = ""; return; }
      let html = `<b>Pipeline:</b> ${p.stage}`;
      if (p.runs && p.runs.length) html += ` · Läufe: ${p.runs.join(", ")}`;
      if (p.champion) html += ` · 🏆 Champion: <b>${p.champion.agent}</b> (Siegquote ${(p.champion.win_rate * 100).toFixed(1)} %, Ø ${p.champion.mean_score.toFixed(1)} Pkt, Turnier ${p.champion.arena})`;
      if (p.error) html += ` · <span style="color:var(--danger)">${p.error}</span>`;
      if (p.log && p.log.length) html += `<div class="muted">${p.log.slice(-3).join(" · ")}</div>`;
      el.innerHTML = html;
    }
    $("#run-select").addEventListener("change", e => { run = e.target.value; compareSel.add(run); tick(); });
    $("#toggle-train-form").addEventListener("click", () => { $("#train-form").hidden = !$("#train-form").hidden; });
    $("#delete-run").addEventListener("click", async () => {
      if (!run) return;
      if (status && status.runs.includes(run)) { toast("Lauf läuft noch – erst stoppen", true); return; }
      if (!confirm(`Lauf „${run}“ mit allen Checkpoints und Logs löschen?`)) return;
      try { await api(`/api/training/runs/${encodeURIComponent(run)}`, { method: "DELETE" }); toast("gelöscht"); compareSel.delete(run); delete compareCache[run]; run = null; await loadRuns(); $("#compare-runs").innerHTML = ""; tick(); }
      catch (e) { toast(e.message, true); }
    });
    $("#toggle-ft-form").addEventListener("click", () => { $("#ft-form").hidden = !$("#ft-form").hidden; });
    $("#start-training").addEventListener("click", async () => {
      try {
        const body = {
          run: $("#tf-run").value, board: $("#tf-board").value, players: +$("#tf-players").value, games: +$("#tf-games").value,
          rounds: +$("#tf-rounds").value, iters: +$("#tf-iters").value, lr: +$("#tf-lr").value, ent: +$("#tf-ent").value,
          margin_weight: +$("#tf-margin").value, eval_every: +$("#tf-eval").value, channels: +$("#tf-channels").value,
          blocks: +$("#tf-blocks").value, resume: $("#tf-resume").value || null, extra: $("#tf-extra").value || null,
          free_rounds: parseInt($("#tf-free-rounds").value, 10) || 0, start_column_always: $("#tf-start-always").checked,
          opponent_view: $("#tf-opp-view").value, algo: $("#tf-algo").value,
          boards: [($("#tf-board-mode").value === "official" ? "official" : ""), $("#tf-boards").value].filter(Boolean).join(",") || null,
          random_boards: +$("#tf-random-boards").value || 0, board_seed: +$("#tf-board-seed").value || 0,
          eval_opponent: $("#tf-eval-opp").value || "heuristic", minibatch: +$("#tf-minibatch").value || null,
        };
        await post("/api/training/start", body);
        $("#train-form").hidden = true; toast("Training gestartet");
        setTimeout(async () => { await loadRuns(); run = body.run; $("#run-select").value = run; tick(); }, 1500);
      } catch (e) { toast(e.message, true); }
    });
    function ftCommon() {
      return {
        iters: +$("#ft-iters").value, players: +$("#ft-players").value, lr: +$("#ft-lr").value,
        kl_ref: +$("#ft-kl").value, l2_ref: +$("#ft-l2").value, warmup_iters: +$("#ft-warmup").value,
        freeze_trunk_iters: +$("#ft-freeze").value, board: $("#ft-board").value, opponent_view: $("#ft-opp-view").value,
      };
    }
    $("#ft-start").addEventListener("click", async () => {
      const bases = Array.from($("#ft-base").selectedOptions).map(o => o.value);
      if (!bases.length) { toast("Basis-Checkpoint wählen", true); return; }
      const c = ftCommon(); const suffix = $("#ft-suffix").value || "-ft"; const usePool = $("#ft-pool").checked;
      let started = 0;
      for (const base of bases) {
        const meta = S.agents.find(a => a.id === base) || {};
        const body = Object.assign({}, c, {
          run: base.split("/")[0] + suffix, finetune_from: base, algo: meta.algo || "ppo",
          opponent_pool: usePool ? bases.filter(b => b !== base) : null,
        });
        try { await post("/api/training/start", body); started++; } catch (e) { toast(`${body.run}: ${e.message}`, true); }
      }
      if (started) { toast(`${started} Finetuning-Lauf/Läufe gestartet`); setTimeout(() => { loadRuns(); tick(); }, 1500); }
    });
    $("#ft-pipeline").addEventListener("click", async () => {
      const bases = Array.from($("#ft-base").selectedOptions).map(o => o.value);
      const c = ftCommon();
      try {
        await post("/api/pipeline/finetune_all", { bases: bases.length ? bases : null, iters: c.iters, players: c.players, use_pool: $("#ft-pool").checked, suffix: $("#ft-suffix").value || "-ft", board: c.board, arena_players: [2, 3], arena_games: 200 });
        toast("Pipeline gestartet: Finetuning → Turnier → Champion"); tick();
      } catch (e) { toast(e.message, true); }
    });
    $("#stop-training").addEventListener("click", async () => {
      try { await post("/api/training/stop" + (status && status.runs.includes(run) ? "?run=" + encodeURIComponent(run) : "")); toast("gestoppt"); tick(); } catch (e) { toast(e.message, true); }
    });
    $$("#range-tabs button").forEach(b => b.addEventListener("click", () => {
      $$("#range-tabs button").forEach(x => x.classList.toggle("active", x === b)); range = +b.dataset.range; render();
    }));

    const f1 = v => v == null || isNaN(v) ? "–" : v.toFixed(1);
    const pct = v => v == null || isNaN(v) ? "–" : (v * 100).toFixed(0) + " %";
    function kpi(label, value, delta, cls, spark, sparkColor) { return { label, value, delta, cls, spark, sparkColor }; }
    function deltaText(cur, prev, suffix) {
      if (cur == null || prev == null || isNaN(cur) || isNaN(prev)) return ["", ""];
      const d = cur - prev; const s = (d >= 0 ? "+" : "") + d.toFixed(1) + (suffix || "");
      return [s, d > 0 ? "up" : d < 0 ? "down" : ""];
    }
    function fmtDur(sec) { if (!isFinite(sec)) return "–"; if (sec < 90) return Math.round(sec) + " s"; if (sec < 5400) return Math.round(sec / 60) + " min"; return (sec / 3600).toFixed(1) + " h"; }

    function renderHeader(meta) {
      const hdr = $("#run-header"); if (!run) { hdr.hidden = true; return; }
      hdr.hidden = false;
      const a = meta.args || {}; const last = records[records.length - 1];
      $("#rh-name").textContent = run;
      const chips = [];
      const chip = (t, hl) => chips.push(`<span class="pill ${hl ? "hl" : ""}">${t}</span>`);
      chip((meta.algo || a.algo || "ppo").toUpperCase(), true);
      chip(`${a.players || "?"} Spieler`);
      chip((meta.opponent_view || a.opponent_view) === "none" ? "blind" : "sieht Gegner");
      const r = meta.rules || {}; chip(r.start_column_always ? "H jederzeit" : "H nur 1. Zug"); chip(`${r.free_rounds || 0} freie Würfe`);
      chip(`Netz ${a.channels || 64}×${a.blocks || 4}` + (meta.params ? ` · ${(meta.params / 1e6).toFixed(2)}M` : ""));
      chip(`${a.games || "?"} Spiele × ${a.rounds || "?"} Runden`);
      chip(`lr ${a.lr}`);
      if (meta.train_boards && meta.train_boards.length > 1) chip(`Blatt-Pool: ${meta.train_boards.length} Blätter`, true);
      if (meta.eval_board) chip(`Eval auf ${meta.eval_board}` + (a.eval_opponent && a.eval_opponent !== "heuristic" ? ` vs ${a.eval_opponent.replace(/\/best\.pt$/, "")}` : ""));
      if (meta.finetuned_from) chip(`Finetuning von ${meta.finetuned_from.replace(/^.*checkpoints\//, "")}`, true);
      if (meta.opponent_pool && meta.opponent_pool.length) chip(`Gegner-Pool: ${meta.opponent_pool.length}`);
      if (a.kl_ref) chip(`KL-Ref ${a.kl_ref}`); if (a.l2_ref && (meta.algo === "dqn")) chip(`L2-Ref ${a.l2_ref}`);
      if (last && last.trunk_frozen) chip("Trunk eingefroren", true);
      $("#rh-chips").innerHTML = chips.join("");
      // progress
      const startIter = records.length ? records[0].iter - 1 : 0;
      const total = startIter + (a.iters || 0);
      const cur = last ? last.iter : startIter;
      const frac = total ? Math.min(1, (cur - startIter) / Math.max(1, total - startIter)) : 0;
      $("#rh-bar").style.width = (frac * 100).toFixed(1) + "%";
      const recent = records.slice(-10); const per = recent.length ? recent.reduce((x, y) => x + y.seconds, 0) / recent.length : NaN;
      const running = status && status.runs.includes(run);
      $("#rh-progress-text").textContent = `Iteration ${cur} von ${total}` + (last ? ` · ${(last.samples / 1e3).toFixed(0)}k Entscheidungen/Iteration · ${last.seconds.toFixed(1)} s/Iteration` : "");
      $("#rh-eta").textContent = running ? `Restzeit ≈ ${fmtDur((total - cur) * per)}` : (cur >= total && total ? "abgeschlossen" : "pausiert");
      // phases
      const noEval = recent.filter(x => x.eval_score == null); const withEval = recent.filter(x => x.eval_score != null);
      const roll = noEval.length ? noEval.reduce((x, y) => x + y.rollout_seconds, 0) / noEval.length : (last ? last.rollout_seconds : 0);
      const upd = noEval.length ? noEval.reduce((x, y) => x + (y.seconds - y.rollout_seconds), 0) / noEval.length : (last ? last.seconds - last.rollout_seconds : 0);
      const evalT = withEval.length && noEval.length ? Math.max(0, withEval.reduce((x, y) => x + y.seconds, 0) / withEval.length - roll - upd) : 0;
      const evalEvery = a.eval_every || 20;
      const parts = [["Rollout (Self-Play auf GPU)", roll, accent], [(meta.algo === "dqn" ? "Replay-Updates" : `${a.algo === "ppo" ? "PPO" : (a.algo || "ppo").toUpperCase()}-Update (${a.epochs || "?"} Ep.)`), upd, blue], [`Eval (alle ${evalEvery} Iter., anteilig)`, evalT / evalEvery, warn]];
      const sum = parts.reduce((x, y) => x + y[1], 0) || 1;
      $("#rh-phases").innerHTML = parts.map(([n, v, c]) => `<div style="width:${(v / sum * 100).toFixed(1)}%;background:${c}" title="${n}: ${v.toFixed(2)} s">${v / sum > 0.12 ? v.toFixed(1) + " s" : ""}</div>`).join("");
      $("#rh-phase-legend").innerHTML = parts.map(([n, v, c]) => `<span><i style="background:${c}"></i>${n}: ${v.toFixed(2)} s</span>`).join("");
    }

    function render() {
      const all = records;
      const recs = range ? all.slice(-range) : all;
      const last = all[all.length - 1];
      const evals = all.filter(r => r.eval_score != null);
      const lastEval = evals[evals.length - 1], prevEval = evals[evals.length - 2];
      const best = evals.reduce((m, r) => r.eval_score > m ? r.eval_score : m, -Infinity);
      const vs = all.filter(r => r.vs_heuristic_win != null);
      const lastVs = vs[vs.length - 1];
      const meta = (runs.find(r => r.run === run) || {}).meta || {};
      const players = meta.args ? meta.args.players : null;
      const isDqn = (meta.algo || (meta.args && meta.args.algo)) === "dqn";
      renderHeader(meta);

      $("#score-headline").textContent = lastEval ? f1(lastEval.eval_score) + " Pkt" : "–";
      const [dt, dc] = deltaText(lastEval && lastEval.eval_score, prevEval && prevEval.eval_score, " vs. letzte Eval");
      const dl = $("#score-delta"); dl.textContent = dt + (baseline ? `  ·  Heuristik ${f1(baseline.heuristic_score)}` : ""); dl.className = "delta " + dc;

      const sparkOf = key => all.slice(-60).map(r => [r.iter, r[key]]);
      const k = [];
      const [d1, c1] = deltaText(last && last.train_score, all[all.length - 11] && all[all.length - 11].train_score, " / 10 Iter.");
      k.push(kpi("Ø Trainings-Score", last ? f1(last.train_score) : "–", d1, c1, sparkOf("train_score"), accent));
      k.push(kpi("Bester Eval-Score", isFinite(best) ? f1(best) : "–", lastEval ? `Iteration ${evals.reduce((m, r) => r.eval_score >= best ? r.iter : m, 0)} → best.pt` : "", "", evals.slice(-60).map(r => [r.iter, r.eval_score]), blue));
      if (players && players > 1) k.push(kpi("Siegquote vs. Heuristik", lastVs ? pct(lastVs.vs_heuristic_win) : "–", lastVs ? `Ø ${f1(lastVs.vs_heuristic_score)} Pkt` : "", lastVs && lastVs.vs_heuristic_win > 0.5 ? "up" : "", vs.slice(-60).map(r => [r.iter, r.vs_heuristic_win]), warn));
      k.push(kpi("Spiellänge (Würfe)", last ? f1(last.train_len) : "–", "", "", sparkOf("train_len"), gray));
      k.push(kpi("Iterationen", last ? last.iter : "0", meta.args ? `von ${meta.args.iters + (meta.args.resume ? "+" : "")}` : "", "", null));
      k.push(kpi("Episoden gesamt", last ? last.episodes_total.toLocaleString("de-DE") : "0", last ? `${last.train_episodes || 0} in letzter Iter.` : "", "", null));
      k.push(kpi("Samples / s", last ? Math.round(last.sps).toLocaleString("de-DE") : "–", last ? `${last.seconds.toFixed(1)} s / Iter.` : "", "", sparkOf("sps"), gray));
      k.push(isDqn ? kpi("Exploration ε", last ? last.entropy.toFixed(3) : "–", last && last.replay ? `Replay ${last.replay.toLocaleString("de-DE")}` : "", "", sparkOf("entropy"), gray)
                   : kpi("Entropie", last ? last.entropy.toFixed(3) : "–", last && last.kl_ref != null ? `KL zur Referenz ${last.kl_ref.toFixed(4)}` : "", "", sparkOf("entropy"), gray));
      const kp = $("#kpis"); kp.innerHTML = "";
      for (const x of k) {
        const d = document.createElement("div"); d.className = "kpi";
        d.innerHTML = `<div class="l">${x.label}</div><div class="v">${x.value}</div><div class="d ${x.cls || ""}">${x.delta || ""}</div>` + (x.spark ? `<div class="spark"></div>` : "");
        kp.appendChild(d);
        if (x.spark) Charts.sparkline(d.querySelector(".spark"), x.spark, x.sparkColor);
      }

      const series = [
        { name: "Ø Score (Training, stochastisch)", color: accent, points: recs.map(r => [r.iter, r.train_score]), fill: true, digits: 2 },
        { name: "Eval-Score (greedy)", color: blue, points: recs.filter(r => r.eval_score != null).map(r => [r.iter, r.eval_score]), dots: true, digits: 2 },
        { name: "neuer Bestwert (best.pt)", color: warn, points: recs.filter(r => r.new_best).map(r => [r.iter, r.eval_score]), dots: true, endDot: false, width: 0, digits: 2 },
      ];
      if (recs.length && baseline) {
        const x0 = recs[0].iter, x1 = recs[recs.length - 1].iter;
        series.push({ name: "Heuristik", color: gray, dashed: true, points: [[x0, baseline.heuristic_score], [x1, baseline.heuristic_score]], digits: 2 });
        series.push({ name: "Zufall", color: red, dashed: true, points: [[x0, baseline.random_score], [x1, baseline.random_score]], digits: 2 });
      }
      Charts.lineChart($("#chart-score"), { series, xLabel: "Iteration" });
      $("#legend-score").innerHTML = series.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");

      const small = isDqn ? [
        ["TD-Loss (Huber)", "value_loss", blue, 4], ["Ø Q-Wert (gewählte Züge)", "q_mean", warn, 3], ["Exploration ε", "entropy", accent, 3],
        ["Spiellänge (Würfe)", "train_len", accent, 1], ["Lernrate", "lr", gray, 6],
      ] : [
        ["Value-Loss", "value_loss", blue, 3], ["Policy-Loss", "policy_loss", warn, 4], ["Entropie", "entropy", accent, 3],
        ["KL-Divergenz (alt→neu)", "kl", red, 4], ["Clip-Anteil", "clipfrac", gray, 3], ["Spiellänge (Würfe)", "train_len", accent, 1],
      ];
      if (all.some(r => r.kl_ref != null)) small.push(["KL zur Referenz-Policy", "kl_ref", purple, 4]);
      if (players && players > 1) small.push(["Siegquote vs. Heuristik", "vs_heuristic_win", warn, 2]);
      small.push(["Samples / s", "sps", gray, 0]);
      const sc = $("#small-charts");
      if (sc.children.length !== small.length || sc.dataset.key !== small.map(x => x[1]).join()) {
        sc.innerHTML = ""; sc.dataset.key = small.map(x => x[1]).join();
        for (const [label] of small) { const c = document.createElement("div"); c.className = "card"; c.innerHTML = `<h2>${label}</h2><div class="cur">–</div><div class="chart"></div>`; sc.appendChild(c); }
      }
      small.forEach(([label, key, color, digits], i) => {
        const c = sc.children[i];
        const pts = recs.filter(r => r[key] != null).map(r => [r.iter, r[key]]);
        const lastV = pts.length ? pts[pts.length - 1][1] : null;
        c.querySelector(".cur").textContent = lastV == null ? "–" : (key === "vs_heuristic_win" ? pct(lastV) : Charts.fmt(lastV, digits));
        Charts.lineChart(c.querySelector(".chart"), { series: [{ name: label, color, points: pts, fill: true, digits, dots: key === "vs_heuristic_win" }], xLabel: "Iteration", yDigits: digits });
      });
    }

    async function renderCompare() {
      const box = $("#compare-runs");
      if (box.children.length !== runs.length) {
        box.innerHTML = "";
        for (const r of runs) {
          const l = document.createElement("label");
          l.innerHTML = `<input type="checkbox" value="${r.run}" ${compareSel.has(r.run) ? "checked" : ""}> ${r.run}`;
          l.querySelector("input").addEventListener("change", e => { if (e.target.checked) compareSel.add(r.run); else compareSel.delete(r.run); renderCompare(); });
          box.appendChild(l);
        }
      }
      const series = []; let i = 0;
      for (const name of Array.from(compareSel)) {
        if (!runs.some(r => r.run === name)) continue;
        const isRunning = status && status.runs.includes(name);
        if (!compareCache[name] || isRunning) {
          try { const r = await api(`/api/training/log?run=${encodeURIComponent(name)}`); compareCache[name] = r.records.filter(x => x.type === "iter"); } catch (e) { continue; }
        }
        const pts = compareCache[name].filter(r => r.eval_score != null).map(r => [r.iter, r.eval_score]);
        series.push({ name, color: PALETTE[i++ % PALETTE.length], points: pts, digits: 2 });
      }
      Charts.lineChart($("#chart-compare"), { series, xLabel: "Iteration" });
      $("#legend-compare").innerHTML = series.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
    }
    window.addEventListener("resize", () => { if (!$("#view-training").hidden) render(); });
    return { show };
  })();

  // ======================================================================= arena
  const Arena = (function () {
    let results = [], timer = null, selected = null;
    const pct = v => v == null ? "–" : (v * 100).toFixed(1) + " %";
    const short = id => id.startsWith("ensemble:") ? id.replace(/^ensemble:(\w+):/, "Ens-$1: ").replace(/\/best\.pt/g, "").replace(/\+/g, "+") : id.replace(/\.pt$/, "");
    async function show() {
      S.agents = await api("/api/agents");
      const box = $("#arena-agents"); box.innerHTML = "";
      for (const a of S.agents) {
        const l = document.createElement("label");
        const on = a.id === "heuristic" || /best\.pt$/.test(a.id) || a.ensemble;
        l.innerHTML = `<input type="checkbox" value="${a.id}" ${on ? "checked" : ""}> ${a.label}` + (a.ensemble ? ` <span class="muted" title="löschen" style="cursor:pointer" data-del="${a.id}">✕</span>` : "");
        if (a.ensemble) l.querySelector("[data-del]").addEventListener("click", async ev => { ev.preventDefault(); await api("/api/ensembles?id=" + encodeURIComponent(a.id), { method: "DELETE" }); show(); });
        box.appendChild(l);
      }
      if (!timer) timer = setInterval(tick, 3000);
      await tick();
    }
    async function tick() {
      if ($("#view-arena").hidden) return;
      try {
        const st = await api("/api/arena/status");
        $("#arena-status").textContent = st.running ? "läuft" : "bereit";
        $("#arena-status").classList.toggle("live", st.running);
        $("#arena-console").textContent = st.tail || "–"; $("#arena-console").scrollTop = 1e9;
        const res = await api("/api/arena/results");
        const changed = res.length !== results.length || (res[0] && results[0] && res[0].id !== results[0].id);
        results = res;
        const sel = $("#arena-select");
        if (changed || !sel.options.length) {
          sel.innerHTML = "";
          for (const r of results) sel.appendChild(new Option(`${r.id}  (${r.result.agents.length} Agenten, ${r.result.games} Spiele/Paarung)`, r.id));
          if (!results.length) sel.appendChild(new Option("– noch kein Turnier –", ""));
          selected = results.length ? results[0].id : null; sel.value = selected || "";
          render();
        }
      } catch (e) { console.warn(e); }
    }
    $("#arena-select").addEventListener("change", e => { selected = e.target.value; render(); });
    $("#ens-add").addEventListener("click", async () => {
      const members = $$("#arena-agents input:checked").map(x => x.value).filter(v => !v.startsWith("ensemble:"));
      if (members.length < 2) { toast("Mindestens 2 Nicht-Ensemble-Agenten markieren", true); return; }
      try {
        const r = await post("/api/ensembles", { members, mode: $("#ens-mode").value });
        toast("Ensemble angelegt");
        await show();
        const cb = $(`#arena-agents input[value="${CSS.escape(r.id)}"]`); if (cb) cb.checked = true;
      } catch (e) { toast(e.message, true); }
    });
    $("#arena-start").addEventListener("click", async () => {
      try {
        const agents = $$("#arena-agents input:checked").map(x => x.value);
        const players = $$("#arena-players input:checked").map(x => +x.value);
        if (agents.length < 2 && players.some(p => p > 1)) { toast("Mindestens 2 Agenten wählen", true); return; }
        await post("/api/arena/start", { agents, players, games: +$("#arena-games").value });
        toast("Turnier gestartet"); tick();
      } catch (e) { toast(e.message, true); }
    });
    function render() {
      const box = $("#arena-results"); box.innerHTML = "";
      const r = results.find(x => x.id === selected); if (!r) return;
      const res = r.result;
      const grid = document.createElement("div"); grid.className = "arena-grid";
      for (const [P, entry] of Object.entries(res.players)) {
        const card = document.createElement("div"); card.className = "card";
        const rows = entry.table.map((t, i) => `<tr><td class="${i === 0 ? "rank1" : ""}">${i + 1}. ${short(t.agent)}</td><td class="num">${t.games}</td><td class="num">${t.win_rate == null ? "–" : `<span class="winbar" style="width:${(t.win_rate * 60).toFixed(0)}px"></span>` + pct(t.win_rate)}</td><td class="num">${t.mean_score.toFixed(2)}</td><td class="num">${t.mean_rank == null ? "–" : t.mean_rank.toFixed(2)}</td></tr>`).join("");
        card.innerHTML = `<h2>${P === "1" ? "Solo (30 Würfe)" : P + " Spieler"}</h2><table class="tbl"><thead><tr><th>Agent</th><th>Spiele</th><th>Siegquote</th><th>Ø Score</th><th>Ø Rang</th></tr></thead><tbody>${rows}</tbody></table>`;
        grid.appendChild(card);
        if (entry.matrix) {
          const names = Object.keys(entry.matrix);
          const m = document.createElement("div"); m.className = "card";
          const head = names.map(n => `<th>${short(n)}</th>`).join("");
          const body = names.map(a => `<tr><td><b>${short(a)}</b></td>` + names.map(b => { const v = entry.matrix[a][b]; const cls = v == null ? "" : v > 0.55 ? "good" : v < 0.45 ? "bad" : ""; return `<td class="${cls}">${a === b ? "·" : pct(v)}</td>`; }).join("") + "</tr>").join("");
          m.innerHTML = `<h2>Direktvergleich (Siegquote Zeile gegen Spalte)</h2><div style="overflow:auto"><table class="tbl matrix"><thead><tr><th></th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
          grid.appendChild(m);
        }
      }
      box.appendChild(grid);
      box.insertAdjacentHTML("beforeend", `<div class="hint" style="margin-top:8px">Regeln: freie Anfangswürfe ${res.free_rounds}, Spalte H jederzeit: ${res.start_column_always ? "ja" : "nein"} · Dauer ${Math.round(res.seconds || 0)} s</div>`);
    }
    return { show };
  })();

  // ======================================================================= eval tab
  const Eval = (function () {
    let results = [], selected = null, timer = null;
    const short = id => id.startsWith("ensemble:") ? id.replace(/^ensemble:(\w+):/, "Ens-$1: ").replace(/\/best\.pt/g, "") : id.replace(/\.pt$/, "");
    const pct = v => v == null ? "–" : (v * 100).toFixed(1) + " %";
    async function show() {
      S.agents = await api("/api/agents");
      const box = $("#eval-agents"); box.innerHTML = "";
      for (const a of S.agents) {
        const l = document.createElement("label");
        l.innerHTML = `<input type="checkbox" value="${a.id}" ${/best\.pt$/.test(a.id) ? "checked" : ""}> ${a.label}`; box.appendChild(l);
      }
      const bb = $("#eval-boards"); bb.innerHTML = "";
      const lo = document.createElement("label"); lo.innerHTML = `<input type="checkbox" value="official" checked> alle offiziellen Blätter (rotierend)`; bb.appendChild(lo);
      for (const b of S.boards) { const l = document.createElement("label"); l.innerHTML = `<input type="checkbox" value="${b.id}" ${b.id === "classic" ? "checked" : ""}> ${b.name}`; bb.appendChild(l); }
      const lr = document.createElement("label"); lr.innerHTML = `<input type="checkbox" value="random"> generierte Zufallsblätter (N, Seed)`; bb.appendChild(lr);
      const eo = $("#eval-opponent"); const prev = eo.value; eo.innerHTML = "";
      for (const a of S.agents) eo.appendChild(new Option(a.label, a.id)); eo.value = prev || "heuristic";
      if (!timer) timer = setInterval(tick, 3000);
      await tick();
    }
    async function tick() {
      if ($("#view-eval").hidden) return;
      try {
        const st = await api("/api/eval/status");
        $("#eval-status").textContent = st.running ? "läuft" : "bereit";
        $("#eval-status").classList.toggle("live", st.running);
        $("#eval-console").textContent = st.tail || "–"; $("#eval-console").scrollTop = 1e9;
        const res = await api("/api/eval/results");
        const changed = res.length !== results.length || (res[0] && results[0] && res[0].id !== results[0].id);
        results = res;
        const sel = $("#eval-select");
        if (changed || !sel.options.length) {
          sel.innerHTML = "";
          for (const r of results) sel.appendChild(new Option(`${r.id}  (${r.result.agents.length} Agenten · ${r.result.board_sets.join(", ")} · ${r.result.players} Sp.)`, r.id));
          if (!results.length) sel.appendChild(new Option("– noch keine Evaluation –", ""));
          selected = results.length ? results[0].id : null; sel.value = selected || "";
          render();
        }
      } catch (e) { console.warn(e); }
    }
    $("#eval-select").addEventListener("change", e => { selected = e.target.value; render(); });
    $("#eval-delete").addEventListener("click", async () => {
      if (!selected || !confirm("Ergebnis löschen?")) return;
      await api(`/api/eval/results/${encodeURIComponent(selected)}`, { method: "DELETE" }); results = []; tick();
    });
    $("#eval-start").addEventListener("click", async () => {
      try {
        const agents = $$("#eval-agents input:checked").map(x => x.value);
        const boards = $$("#eval-boards input:checked").map(x => x.value === "random" ? `random:${+$("#eval-random-n").value || 8}:${+$("#eval-random-seed").value || 0}` : x.value);
        if (!agents.length || !boards.length) { toast("Agenten und Datensätze wählen", true); return; }
        await post("/api/eval/start", {
          agents, boards, players: +$("#eval-players").value, games: +$("#eval-games").value, seed: +$("#eval-seed").value,
          opponent: $("#eval-opponent").value, start_column_always: $("#eval-start-always").checked,
          free_rounds: +$("#eval-free-rounds").value || 0, label: $("#eval-label").value || null,
        });
        toast("Evaluation gestartet"); tick();
      } catch (e) { toast(e.message, true); }
    });
    function render() {
      const box = $("#eval-results"); box.innerHTML = "";
      const r = results.find(x => x.id === selected); if (!r) return;
      const res = r.result; const sets = res.board_sets;
      const grid = document.createElement("div"); grid.className = "eval-grid";
      const bySet = {};
      for (const row of res.results) (bySet[row.spec] = bySet[row.spec] || []).push(row);
      for (const spec of sets) {
        const rows = (bySet[spec] || []).slice().sort((a, b) => b.eval_score - a.eval_score);
        const maxScore = Math.max(...rows.map(x => x.eval_score), 1);
        const card = document.createElement("div"); card.className = "card";
        const label = rows[0] ? rows[0].board_set : spec;
        const hasVs = rows.some(x => x.vs_heuristic_win != null);
        card.innerHTML = `<h2>Datensatz: ${label} · ${res.players} Spieler · ${res.games} Spiele je Agent · Würfel-Seed ${res.seed}${hasVs ? ` · Referenzgegner ${short(res.opponent || "heuristic")}` : ""}</h2>
          <table class="tbl"><thead><tr><th>Agent</th><th>Ø Score (Self-Play, greedy)</th><th>± Std</th><th>Ø Länge</th>${hasVs ? "<th>Siegquote vs. Referenz</th><th>Ø Score vs. Referenz</th>" : ""}</tr></thead><tbody>` +
          rows.map((x, i) => `<tr><td class="${i === 0 ? "rank1" : ""}">${i + 1}. ${short(x.agent)}</td><td class="num"><span class="bar-cell"><i style="width:${Math.max(0, x.eval_score / maxScore * 80).toFixed(0)}px"></i>${x.eval_score.toFixed(2)}</span></td><td class="num">${(x.eval_score_std || 0).toFixed(2)}</td><td class="num">${x.eval_len.toFixed(1)}</td>${hasVs ? `<td class="num">${pct(x.vs_heuristic_win)}</td><td class="num">${x.vs_heuristic_score == null ? "–" : x.vs_heuristic_score.toFixed(2)}</td>` : ""}</tr>`).join("") +
          `</tbody></table>`;
        grid.appendChild(card);
      }
      box.appendChild(grid);
      box.insertAdjacentHTML("beforeend", `<div class="hint" style="margin-top:8px">Regeln: freie Anfangswürfe ${res.free_rounds}, Spalte H jederzeit: ${res.start_column_always ? "ja" : "nein"} · gleiche Würfel-Seeds für alle Agenten · Dauer ${Math.round(res.seconds || 0)} s</div>`);
    }
    return { show };
  })();

  init().catch(e => toast(e.message, true));
})();
