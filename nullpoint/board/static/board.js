// Nullpoint board client: renders server state, sends keyboard fallbacks.
(() => {
  const $ = (id) => document.getElementById(id);
  let ws, last = null, audio = null, explain = false;
  const trail = { plain: [], adapt: [] };          // recent gaze positions for the strip

  function tick(freq = 660, dur = 0.06, vol = 0.08) {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = freq; g.gain.value = vol;
      o.connect(g); g.connect(audio.destination);
      o.start(); g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + dur); o.stop(audio.currentTime + dur);
    } catch (e) { /* audio is optional */ }
  }
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function layout(meta) {
    // focus card sits at the null-point position; side zones fill the rest
    const x = meta.null_x ?? 0.5;
    const focus = $("focus");
    focus.style.left = (x * 100) + "%";
    const half = Math.min(0.22, x - 0.05, 0.95 - x);
    $("z-L").style.width = Math.max(0, (x - half - 0.02)) * 100 + "%";
    $("z-R").style.width = Math.max(0, (1 - x - half - 0.02)) * 100 + "%";
    $("null-tag").style.display = (meta.null_deg !== null && meta.adapt) ? "" : "none";
    if (meta.null_deg !== null) $("null-tag").textContent = `steadiest gaze · ${meta.null_deg > 0 ? "+" : ""}${meta.null_deg.toFixed(1)}°`;
  }

  // ------------------------------------------------------------------ overlays: setup, null zone, phrase test
  const CARDS = {
    setup: ["Quick setup · 10 seconds", "Look at each dot until it moves on",
            "<p>This tunes the board to how you are sitting today. Nothing is recorded.</p>"],
    nullfind: ["Find your null zone · 40 seconds", "Where are your eyes calmest?",
               "<p>Three short steps: follow dots with your eyes, point your nose at two dots, then read some small words. Nothing is recorded.</p>"],
    task: ["Phrase test · about 8 minutes", "Find the phrase shown at the top",
           "<p>4 blocks of 6 phrases. Move with your eyes and blink firmly to select, as quickly and accurately as you can. " +
           "Adaptation switches on or off between blocks; the screen won't say which.</p>"],
  };
  const SAY = {
    eyes: "Keep your head still, facing the screen. Follow the dot with your <b>eyes</b>.",
    head: "Now point your <b>nose</b> at the dot.",
    read: "Read the words. Hold your head <b>however makes them easiest to see</b>.",
  };
  const GO_START = "<b>Blink firmly</b> or press <kbd>Space</kbd> to start · <kbd>Esc</kbd> to skip";

  function taskResults(t) {
    const row = (name, r) => r ? `<tr><td>${name}</td><td>${r.found} of ${r.phrases}</td><td>${r.median_time_s ?? "–"} s</td>` +
      `<td>${r.wrong_selections_per_phrase}</td><td>${r.extra_steps_per_phrase}</td></tr>` : "";
    const s = t.summary || {};
    return `<table class="res"><tr><th></th><th>found</th><th>median time</th><th>wrong selections</th><th>extra steps</th></tr>` +
      row("Adaptation off", s.off) + row("Adaptation on", s.on) + `</table>` +
      `<p class="note">${t.sim_amp ? `With ${t.sim_amp}° simulated nystagmus. ` : ""}Per phrase. Saved to ${esc(t.saved || "reports/")}.</p>`;
  }

  function renderOverlay(s, prevState) {
    const su = s.setup, nf = s.nullfind, tk = s.task;
    const sp = su && su.phase, np = nf && nf.phase, tp = tk && tk.phase;
    const active = np && np !== "done" ? "nullfind" : (sp && sp !== "done" ? "setup" : (tp === "ready" || tp === "done" ? "task" : null));
    const cur = { nullfind: nf, setup: su, task: tk }[active];
    const prev = prevState && active ? prevState[active] : null;
    $("setup").hidden = !active;
    const card = active && (cur.phase === "ready" || (active === "task" && cur.phase === "done"));
    $("setup-card").style.display = card ? "" : "none";
    if (card) {
      if (active === "task" && cur.phase === "done") {
        $("sc-title").textContent = "Phrase test · results"; $("sc-head").textContent = "Adaptation off vs on";
        $("sc-body").innerHTML = taskResults(cur); $("sc-go").innerHTML = "<b>Blink</b> or press <kbd>Esc</kbd> to close";
      } else {
        [$("sc-title").textContent, $("sc-head").textContent] = CARDS[active];
        $("sc-body").innerHTML = CARDS[active][2]; $("sc-go").innerHTML = GO_START;
      }
    }
    selfView(card && active !== "task" && cur.phase === "ready", s);
    const dotPhase = active && ["run", "eyes", "head"].includes(cur.phase);
    const dot = $("tdot");
    dot.classList.toggle("show", !!dotPhase);
    $("nf-say").innerHTML = active === "nullfind" && SAY[cur.phase] ? SAY[cur.phase] : "";
    const word = $("nf-word");
    word.textContent = active === "nullfind" && cur.phase === "read" ? cur.word : "";
    if (active === "nullfind" && cur.phase === "read") word.style.fontSize = cur.size + "px";
    if (dotPhase) {
      dot.style.left = (cur.x * 100) + "%";
      $("ring").style.strokeDashoffset = 326.7 * (1 - cur.p);
    }
    if (active && active !== "task" && cur.phase !== "ready") {
      $("setup-step").textContent = active === "setup" ? `Look at the dot · ${cur.i + 1} of ${cur.n}` : `Step ${cur.i + 1} of ${cur.n}`;
      if (!prev || prev.i !== cur.i) tick(560 + 30 * cur.i, 0.07);
    } else {
      $("setup-step").textContent = "";
    }
    const done = np === "done" ? nf : (sp === "done" ? su : null);
    const toast = $("toast");
    toast.hidden = !done;
    if (done) {
      toast.textContent = done.msg; toast.classList.toggle("bad", !done.ok);
      const pd = prevState && (np === "done" ? prevState.nullfind : prevState.setup);
      if (!pd || pd.phase !== "done") tick(done.ok ? 880 : 300, 0.15, 0.1);
    }
  }

  // camera self-view on the setup / null-zone cards, so people can see they're in frame
  let svTimer = null;
  function selfView(on, s) {
    const sv = $("sv");
    if (!on || (last && last.meta.mode === "sim")) { sv.hidden = true; clearInterval(svTimer); svTimer = null; return; }
    sv.hidden = false;
    $("sv-face").textContent = s.face ? "✓ face found" : "no face yet: sit in front of the camera, light on your face";
    $("sv-face").className = s.face ? "ok" : "warn";
    if (!svTimer) svTimer = setInterval(() => { $("sv-img").src = "/preview.jpg?t=" + Date.now(); }, 250);
  }
  $("sv-img").onerror = () => { $("sv").hidden = true; };

  // ------------------------------------------------------------------ phrase-test banner
  function renderTask(tk, prevTk) {
    const on = tk && ["block", "trial", "gap"].includes(tk.phase);
    $("task-banner").hidden = !on;
    if (!on) return;
    if (tk.phase === "block") {
      $("tb-main").textContent = `Block ${tk.block} of ${tk.blocks}`; $("tb-sub").textContent = "get ready · eyes on the card";
    } else if (tk.phase === "trial") {
      $("tb-main").innerHTML = `Find: <b>${esc(tk.target)}</b>`; $("tb-sub").textContent = `block ${tk.block} of ${tk.blocks} · phrase ${tk.trial} of ${tk.trials}`;
      if (!prevTk || prevTk.phase !== "trial") tick(700, 0.05);
    } else {
      $("tb-main").textContent = tk.last === "found" ? "✓ found" : "time's up"; $("tb-sub").textContent = "";
    }
  }

  // ------------------------------------------------------------------ gaze strip (explain view)
  function renderStrip(s) {
    const show = !!s.strip && (explain || !!s.sim);
    $("gstrip").hidden = !show;
    if (!show) { trail.plain = []; trail.adapt = []; return; }
    trail.plain.push(s.strip.plain); trail.adapt.push(s.strip.adapt);
    for (const k of ["plain", "adapt"]) if (trail[k].length > 24) trail[k].shift();
    $("gs-active").textContent = s.task && s.task.phase !== "done" ? "" : `${s.adapt ? "Nullpoint" : "ordinary smoothing"} drives the board`;
    const cv = $("gcanvas"), dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = 46;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    const [b1, b2] = s.strip.bounds;
    const seg = (x0, x1, fill, label) => {
      c.fillStyle = fill; c.fillRect(x0 * W, 0, (x1 - x0) * W, H);
      c.fillStyle = "rgba(154,166,178,.55)"; c.font = "11px Segoe UI, system-ui, sans-serif";
      c.fillText(label, x0 * W + 6, 12);
    };
    seg(0, b1, "rgba(242,181,68,.06)", "LEFT"); seg(b1, b2, "rgba(79,179,165,.07)", "CARD"); seg(b2, 1, "rgba(242,181,68,.06)", "RIGHT");
    c.strokeStyle = "rgba(154,166,178,.35)"; c.setLineDash([4, 4]);
    for (const b of [b1, b2]) { c.beginPath(); c.moveTo(b * W, 0); c.lineTo(b * W, H); c.stroke(); }
    c.setLineDash([]);
    const row = (arr, y, rgb) => {
      arr.forEach((x, i) => {
        c.fillStyle = `rgba(${rgb},${0.08 + 0.5 * i / arr.length})`;
        c.beginPath(); c.arc(x * W, y, 3, 0, 2 * Math.PI); c.fill();
      });
      const x = arr[arr.length - 1];
      c.fillStyle = `rgb(${rgb})`; c.beginPath(); c.arc(x * W, y, 7, 0, 2 * Math.PI); c.fill();
    };
    row(trail.plain, 20, "154,166,178"); row(trail.adapt, 36, "242,181,68");
  }

  // ------------------------------------------------------------------ main render
  function render(msg) {
    const s = msg.state, m = msg.meta;
    const testing = s.task && ["block", "trial", "gap"].includes(s.task.phase);
    $("c-mode").textContent = m.mode === "sim" ? "simulation (keyboard)" : "live camera";
    $("c-face").textContent = m.mode === "sim" ? "no camera" : (s.face ? "face tracked" : "no face");
    $("c-face").className = "chip " + (m.mode === "sim" ? "" : (s.face ? "on" : "warn"));
    $("c-sim").hidden = !s.sim;
    if (s.sim) $("c-sim").textContent = `simulated nystagmus · ${s.sim.amp}° ${s.sim.kind} ${s.sim.freq} Hz`;
    $("c-adapt").textContent = testing ? "test running" : s.adapt
      ? (m.freq_hz ? `adapted · ${m.freq_hz.toFixed(1)} Hz${s.sim ? " (simulated profile)" : ""}` : "adapt on · no profile") : "adapt off";
    $("c-adapt").className = "chip " + (!testing && s.adapt && m.freq_hz ? "on" : "");
    $("c-fps").textContent = m.mode === "sim" ? "—" : `${m.fps} fps`;
    $("c-cal").hidden = !m.adaptable;
    $("c-cal").textContent = s.setup_ok ? `set up · ${s.corrections} self-correction${s.corrections === 1 ? "" : "s"}` : "not set up · K";
    $("c-cal").className = "chip " + (s.setup_ok ? "on" : "");
    layout(m);
    renderOverlay(s, last && last.state);
    renderTask(s.task, last && last.state.task);
    renderStrip(s);

    const tiles = s.tiles;
    const strip = $("strip");
    if (strip.children.length !== tiles.length || !last || last.state.page !== s.page || last.state.tiles.join() !== tiles.join()) {
      strip.innerHTML = tiles.map((t, i) => `<div class="tile${t === "More options" ? " more" : ""}" data-i="${i}">${esc(t)}</div>`).join("");
    }
    [...strip.children].forEach((el, i) => el.classList.toggle("cur", i === s.cursor));
    $("phrase").textContent = tiles[s.cursor];
    $("page").textContent = s.page_name;
    $("said").innerHTML = s.last_spoken ? `Last said: <b>${esc(s.last_spoken)}</b>` : "";
    ["L", "R"].forEach((z) => $("z-" + z).classList.toggle("active", s.zone === z));
    s.belief.forEach((b, i) => $("b" + i).style.width = (b * 100) + "%");
    $("z-L").querySelector(".fill").style.height = (s.belief[0] * 100) + "%";
    $("z-R").querySelector(".fill").style.height = (s.belief[2] * 100) + "%";
    $("alarm").classList.toggle("show", s.emergency);
    $("pending").hidden = !s.alarm_pending;

    if (last) {
      if (last.state.cursor !== s.cursor || last.state.page !== s.page) tick(620, 0.05);
      if (s.history.length !== last.state.history.length) {
        tick(880, 0.12, 0.1);
        $("focus").classList.add("flash"); setTimeout(() => $("focus").classList.remove("flash"), 450);
      }
      if (s.alarm_pending && !last.state.alarm_pending) { tick(520, 0.25, 0.06); setTimeout(() => tick(660, 0.3, 0.06), 260); }
      if (s.emergency && !last.state.emergency) tick(300, 0.5, 0.2);
    }
    last = msg;
  }

  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = (e) => render(JSON.parse(e.data));
    ws.onclose = () => { $("c-mode").textContent = "reconnecting…"; setTimeout(connect, 1000); };
  }
  const send = (cmd) => ws && ws.readyState === 1 && ws.send(JSON.stringify({ cmd }));

  document.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (!$("help").hidden && (k === "h" || k === "escape" || k === "?")) { $("help").hidden = true; return; }
    if (k === "arrowleft") send("left");
    else if (k === "arrowright") send("right");
    else if (k === " " || k === "enter") { e.preventDefault(); send("blink"); }
    else if (k === "e") send("emergency");
    else if (k === "c") send("clear");
    else if (k === "r") send("speak_again");
    else if (k === "k") send("setup");
    else if (k === "n") send("nullzone");
    else if (k === "t") send("task");
    else if (k === "s") send("sim");
    else if (k === "a") send("adapt");
    else if (k === "d") { explain = !explain; if (last) renderStrip(last.state); }
    else if (k === "h" || k === "?") { $("help").hidden = false; caregiverInfo(); }
    else if (k === "escape") send("setup_skip");
  });
  async function caregiverInfo() {
    try {
      const j = await (await fetch("/pin")).json();
      $("cg-info").innerHTML = (j.caregiver_url ? `Phone: <b>${esc(j.caregiver_url)}</b> (same Wi-Fi). ` : "Start with <code>--lan</code> to open the caregiver view on a phone. ") +
        `PIN for editing phrases: <b>${esc(j.pin)}</b>`;
    } catch (e) { /* not on the laptop */ }
  }
  // demo / screenshot switches: ?explain=1 shows the gaze strip, ?help=1 opens the help screen
  const q = new URLSearchParams(location.search);
  if (q.has("explain")) explain = true;
  if (q.has("help")) { $("help").hidden = false; caregiverInfo(); }
  connect();
})();
