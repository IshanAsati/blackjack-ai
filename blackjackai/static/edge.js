const $ = id => document.getElementById(id);
const format = new Intl.NumberFormat("en-US");
const signed = (n, decimals = 2) => `${n >= 0 ? "+" : ""}${n.toFixed(decimals)}`;
const pct = n => n === null || n === undefined ? "—" : `${signed(n * 100)}%`;
let state = null, evidence = null, training = null;
let busy = false, running = false, stopping = false, target = null;
let toastTimer;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "The table could not complete this action.");
  return payload;
}
function storageGet(key) { try { return sessionStorage.getItem(key); } catch { return null; } }
function storageSet(key, value) { try { sessionStorage.setItem(key, value); } catch {} }
function toast(message) { $("toast").textContent = message; $("toast").classList.add("visible"); clearTimeout(toastTimer); toastTimer = setTimeout(() => $("toast").classList.remove("visible"), 6000); }
function controls() {
  for (const id of ["next-round", "next-bet", "benchmark", "new-session", "seed"]) $(id).disabled = busy || running || !state;
  $("auto").disabled = !state || (busy && !running) || stopping;
  $("auto").textContent = stopping ? "Pausing…" : running ? "Pause auto" : "Start auto";
  $("auto").classList.toggle("is-running", running);
  $("running-dot").classList.toggle("running", running);
  $("run-status").textContent = stopping ? "Finishing this batch…" : running ? (target ? `Auto · ${format.format(Math.max(0, target - state.rounds))} rounds left` : "Auto running") : busy ? "Playing batch…" : "Paused";
  $("export").disabled = !state || busy;
}
function cards(container, values) {
  const fragment = document.createDocumentFragment();
  for (const value of values) {
    const card = document.createElement("div"); card.className = "card"; card.setAttribute("aria-label", value === 1 ? "Ace" : `Card value ${value}`);
    for (let i = 0; i < 2; i++) { const rank = document.createElement("span"); rank.textContent = value === 1 ? "A" : value; card.append(rank); }
    fragment.append(card);
  }
  $(container).replaceChildren(fragment);
}
function render(next) {
  state = next; storageSet("edge-session", next.id); $("seed").value = next.seed;
  const headline = (next.sizing && next.sizing !== "flat 1 unit" && next.sized_profit !== undefined) ? next.sized_profit : next.profit;
  $("session-profit").innerHTML = `${signed(headline, 1)} <small>units</small>`;
  $("session-profit").style.color = headline < 0 ? "#c94732" : "#17313a";
  $("round-count").textContent = format.format(next.rounds); $("bet-count").textContent = format.format(next.bets); $("skip-count").textContent = format.format(next.skipped);
  const decision = next.next;
  const stake = decision.stake_units || 0;
  $("next-decision").textContent = decision.action === "bet" ? (stake > 1 ? `Bet ${stake} units (Kelly).` : "Bet one unit.") : "Sit this one out.";
  $("decision-copy").textContent = decision.action === "bet" ? `Sized by half-Kelly against a 400-unit roll (${next.sizing || "flat"}). It commits before the next deal.` : "The model waits while the table keeps playing. Every exposed card updates the count.";
  $("true-count").textContent = signed(decision.true_count); $("predicted-return").textContent = pct(decision.estimated_return); $("bet-threshold").textContent = pct(next.threshold);
  $("cards-left").textContent = `${decision.cards_remaining} cards left`; $("shoe-label").textContent = `Shoe ${format.format(decision.shoe)}`;
  $("shoe-fill").style.width = `${decision.fraction_remaining * 100}%`; $("shoe-fill").parentElement.setAttribute("aria-valuenow", decision.cards_remaining);
  const last = next.last_round;
  if (last) {
    $("table-heading").textContent = `Round ${format.format(last.round)} · complete`;
    $("last-bet").textContent = last.selected ? `${signed(last.sized_profit ?? last.profit, 1)} units · ${last.stake || 1}u bet placed` : "Observed · no wager";
    $("player-label").textContent = last.selected ? "Our hand" : "Table player";
    if (last.split && last.hands) {
      const frag = document.createDocumentFragment();
      last.hands.forEach((hand, i) => {
        const wrap = document.createElement("div"); wrap.className = "split-hand";
        const tag = document.createElement("span"); tag.className = "split-tag"; tag.textContent = `Hand ${i + 1} · ${hand.total}`;
        const row = document.createElement("div"); row.className = "split-cards";
        for (const value of hand.cards) {
          const card = document.createElement("div"); card.className = "card";
          for (let k = 0; k < 2; k++) { const rank = document.createElement("span"); rank.textContent = value === 1 ? "A" : value; card.append(rank); }
          row.append(card);
        }
        wrap.append(tag, row); frag.append(wrap);
      });
      $("player-cards").replaceChildren(frag);
    } else {
      cards("player-cards", last.player);
    }
    cards("dealer-cards", last.dealer);
    $("player-total").textContent = last.split ? last.hands.map(h => h.total).join(" / ") : last.player_total;
    $("dealer-total").textContent = last.dealer_total;
    const moves = last.split ? last.hands.map((h, i) => `H${i + 1} ${h.moves.join("→")}`).join(" · ") : (last.moves.length ? last.moves.join(" → ") : last.natural ? "Natural blackjack" : "Dealer blackjack");
    $("round-note").textContent = `${moves}. Pre-deal count ${signed(last.predeal_true_count)}. ${last.selected ? "Result included in net profit." : "No gain or loss while observing."}`;
  } else {
    $("table-heading").textContent = "Watch the next hand."; $("last-bet").textContent = "No hand dealt";
    $("player-label").textContent = "Table player"; cards("player-cards", []); cards("dealer-cards", []);
    $("player-total").textContent = $("dealer-total").textContent = "—";
    $("round-note").textContent = "Bets are chosen before the deal. Cards are revealed after the hand.";
  }
  $("live-roi").textContent = pct(next.return_per_initial_unit); $("baseline-profit").textContent = `${signed(next.baseline_profit, 1)} u`;
  $("results").textContent = `${format.format(next.wins)} / ${format.format(next.losses)} / ${format.format(next.pushes)}`;
  $("live-interval").textContent = next.completed_shoes_95ci ? `Completed-shoe return ${pct(next.completed_shoes_return)}; approximate 95% range ${pct(next.completed_shoes_95ci[0])} to ${pct(next.completed_shoes_95ci[1])}. Live descriptive estimate, not a fixed-length test.` : "Session uncertainty appears after 100 completed shoes and 30 shoes with bets. Short sessions can lose.";
  drawProfit(); controls();
}

function plot(id, series, {width, height, reference = null, selectedX = null, percent = false}) {
  width = Math.max(300, Math.round($(id).clientWidth) || width);
  $(id).setAttribute("viewBox", `0 0 ${width} ${height}`);
  const pad = {left: 62, right: 20, top: 24, bottom: 34};
  const points = series.flatMap(s => s.points); if (!points.length) return;
  let xmin = 0, xmax = Math.max(1, ...points.map(p => p[0]));
  let ys = points.map(p => p[1]); if (reference !== null) ys.push(reference); if (!percent) ys.push(0);
  let ymin = Math.min(...ys), ymax = Math.max(...ys); const margin = Math.max((ymax - ymin) * .13, percent ? .001 : 1);
  ymin -= margin; ymax += margin;
  const x = v => pad.left + v / xmax * (width - pad.left - pad.right);
  const y = v => height - pad.bottom - (v - ymin) / (ymax - ymin) * (height - pad.top - pad.bottom);
  let svg = "";
  for (let i = 0; i <= 4; i++) {
    const value = ymin + (ymax - ymin) * i / 4, yy = y(value);
    svg += `<line x1="${pad.left}" x2="${width-pad.right}" y1="${yy}" y2="${yy}" stroke="#d9d3c8"/><text x="${pad.left-9}" y="${yy+4}" text-anchor="end" fill="#607376" font-size="11" font-family="Consolas,monospace">${percent ? (value * 100).toFixed(1) + "%" : Math.abs(value) >= 1000 ? (value / 1000).toFixed(1) + "k" : value.toFixed(0)}</text>`;
    const xx = x(xmax * i / 4);
    svg += `<text x="${xx}" y="${height-9}" text-anchor="middle" fill="#607376" font-size="11" font-family="Consolas,monospace">${Math.round(xmax * i / 4).toLocaleString("en-US")}</text>`;
  }
  if (!percent) svg += `<line x1="${pad.left}" x2="${width-pad.right}" y1="${y(0)}" y2="${y(0)}" stroke="#99a59c" stroke-dasharray="3 4"/>`;
  if (reference !== null) svg += `<line x1="${pad.left}" x2="${width-pad.right}" y1="${y(reference)}" y2="${y(reference)}" stroke="#8a743b" stroke-dasharray="6 4"/>`;
  if (selectedX !== null) svg += `<line x1="${x(selectedX)}" x2="${x(selectedX)}" y1="${pad.top}" y2="${height-pad.bottom}" stroke="#889893" stroke-dasharray="3 5"/>`;
  for (const s of series) {
    svg += `<polyline points="${s.points.map(p => `${x(p[0])},${y(p[1])}`).join(" ")}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linejoin="round"/>`;
    if (percent) for (const p of s.points) svg += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="${p[0] === selectedX ? 5 : 3}" fill="${s.color}"><title>Episode ${p[0]}: ${(p[1] * 100).toFixed(3)}%</title></circle>`;
  }
  $(id).innerHTML = svg;
}
function drawProfit() {
  if (!state) return;
  const series = [{color: "#277664", points: state.history.map(p => [p.round, p.profit])}];
  if ($("show-baseline").checked) series.push({color: "#c94732", points: state.history.map(p => [p.round, p.baseline])});
  plot("profit-chart", series, {width: 650, height: 270});
}

async function manual(rounds, untilBet = false) {
  if (busy || running || !state) return; busy = true; controls();
  const previousBets = state.bets;
  try { render(await api("/api/edge/advance", {id: state.id, rounds, until_bet: untilBet})); if (untilBet && state.bets === previousBets) toast("No qualifying bet in these 10,000 rounds. All rounds were recorded."); }
  catch (error) { toast(error.message); }
  finally { busy = false; controls(); }
}
async function runAuto(fixedRounds = null) {
  if (busy || running || !state) return;
  running = true; stopping = false; target = fixedRounds ? state.rounds + fixedRounds : null; controls();
  $("control-note").textContent = fixedRounds ? "Fixed run in progress. All outcomes count toward this session." : "Auto is running. Pause keeps every result and finishes the current batch.";
  try {
    while (!stopping && (!target || state.rounds < target)) {
      const speed = Number($("speed").value), remaining = target ? target - state.rounds : Infinity;
      const amount = Math.min(target ? 1000 : speed, remaining);
      busy = true; controls();
      render(await api("/api/edge/advance", {id: state.id, rounds: amount}));
      busy = false; controls();
      if (!stopping && (!target || state.rounds < target)) await new Promise(resolve => setTimeout(resolve, target ? 30 : speed === 1 ? 1000 : 200));
    }
    $("control-note").textContent = stopping ? "Paused. Every completed round remains in this session." : "Fixed run complete. Results include every round, whether profitable or not.";
  } catch (error) { toast(error.message); $("control-note").textContent = "Auto stopped after a request error. Reload to restore the server’s session before continuing."; }
  finally { busy = running = stopping = false; target = null; controls(); }
}
async function newSession() {
  if (busy || running) return;
  const raw = $("seed").value.trim(); const body = raw ? {seed: Number(raw)} : {};
  if (raw && (!Number.isSafeInteger(body.seed) || body.seed < 0 || body.seed >= 2**32)) return toast("Use a whole seed between 0 and 4,294,967,295.");
  busy = true; controls();
  try { render(await api("/api/edge/new", body)); $("control-note").textContent = "New session. Auto plays continuously until paused."; }
  catch (error) { toast(error.message); }
  finally { busy = false; controls(); }
}

function renderTraining() {
  if (!training || !training.runs.length) return;
  const longest = training.runs.reduce((a,b) => a.points.length > b.points.length ? a : b);
  $("checkpoint").max = Math.max(0, longest.points.length - 1);
  const idx = Math.min(Number($("checkpoint").value), longest.points.length - 1), episode = longest.points[idx].episode;
  $("checkpoint-label").textContent = `${format.format(episode)} episodes`;
  plot("training-chart", training.runs.map((run,i) => ({color: i ? "#c94732" : "#277664", points: run.points.map(p => [p.episode,p.exact_mean_reward])})), {width:800, height:330, reference:training.optimal_return, selectedX:episode, percent:true});
  $("checkpoint-values").replaceChildren();
  for (const run of training.runs) {
    const p = run.points.find(p => p.episode === episode) || run.points.at(-1), div = document.createElement("div"), heading = document.createElement("p"), value = document.createElement("strong"), detail = document.createElement("p");
    heading.textContent = run.name; value.textContent = pct(p.exact_mean_reward); detail.textContent = `Random exploration: ${(p.epsilon * 100).toFixed(1)}% · ${p.action_mismatches} recorded decision mismatches`;
    div.append(heading,value,detail); $("checkpoint-values").append(div);
  }
}
async function loadTraining() {
  try {
    training = await api("/api/training");
    if (training.betting) $("bet-training-info").textContent = `${format.format(training.betting.training_rounds)} training rounds. The model learned betting values from public counts and simulated outcomes. Separate validation chose a ${pct(training.betting.selected_threshold)} threshold before testing.`;
    if (!training.runs.length) { $("training-cards").textContent = "No saved training history found."; return; }
    $("training-cards").replaceChildren();
    for (const run of training.runs) {
      const card = document.createElement("div"); card.className = "training-run";
      const title = document.createElement("h3"), value = document.createElement("p"), note = document.createElement("p"); value.className = "training-value";
      title.textContent = run.name; value.textContent = run.report ? pct(run.report.best_checkpoint.exact_mean_reward) : "In progress";
      note.textContent = `Best retained policy · ${format.format(run.points.at(-1).episode)} episodes · ${run.status}`;
      card.append(title,value,note); $("training-cards").append(card);
    }
    if (training.unsupervised) {
      const run = training.unsupervised, card = document.createElement("div"); card.className = "training-run";
      const title = document.createElement("h3"), note = document.createElement("p"); title.textContent = "Unsupervised representation";
      note.textContent = `${format.format(run.unlabeled_states)} states · ${run.epochs} epochs. ${(100 * run.player_total_reconstruction_accuracy).toFixed(1)}% total reconstruction on training data. No standalone playing policy.`;
      card.append(title,note); $("training-cards").append(card);
    }
    if (training.count) {
      const c = training.count, card = document.createElement("div"); card.className = "training-run";
      const title = document.createElement("h3"), value = document.createElement("p"), note = document.createElement("p"); value.className = "training-value";
      title.textContent = "Count-aware neural strategist";
      value.textContent = `${(100 * c.distilled.grid_agreement).toFixed(1)}% analytical agreement`;
      const parts = [];
      if (c.rl_from_scratch) parts.push(`RL from scratch ${pct(c.rl_from_scratch.test_neural.profit_per_hand)}/hand vs fixed ${pct(c.rl_from_scratch.test_fixed_strategy.profit_per_hand)}/hand`);
      if (c.rl_fine_tuned) parts.push(`RL fine-tune ${pct(c.rl_fine_tuned.test_neural.profit_per_hand)}/hand`);
      note.textContent = `Distilled play net (hit/stand/double from true count) now deals the live table. ${parts.join("; ") || "No RL play runs recorded."} Play alone is about break-even; the betting model carries the edge.`;
      card.append(title,value,note); $("training-cards").append(card);
    }
    $("checkpoint").value = String(Math.max(...training.runs.map(r => r.points.length)) - 1); renderTraining();
  } catch (error) { toast(error.message); }
}
function switchView(view) {
  for (const name of ["live", "training", "sim"]) $(`${name}-view`).hidden = name !== view;
  for (const [id, active] of [["training-tab", view === "training"], ["live-tab", view === "live"], ["sim-tab", view === "sim"]]) { $(id).classList.toggle("selected", active); $(id).setAttribute("aria-pressed", active); }
  if (view === "training") { if (!training) loadTraining(); else renderTraining(); } else if (view === "live") drawProfit();
}
$("live-tab").addEventListener("click", () => switchView("live")); $("training-tab").addEventListener("click", () => switchView("training")); $("sim-tab").addEventListener("click", () => switchView("sim"));
$("checkpoint").addEventListener("input",renderTraining); $("show-baseline").addEventListener("change",drawProfit);
$("next-round").addEventListener("click", () => manual(1)); $("next-bet").addEventListener("click", () => manual(10000,true));
$("auto").addEventListener("click", () => { if (running) { stopping = true; controls(); } else runAuto(); });
$("benchmark").addEventListener("click", () => runAuto(100000)); $("new-session").addEventListener("click",newSession);
$("export").addEventListener("click", () => { if (!state) return; const blob = new Blob([JSON.stringify({session:state, previous_fixed_test:evidence},null,2)],{type:"application/json"}); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = `blackjack-session-${state.seed}-${state.rounds}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000); });

let simRunning = false;
const simDollars = n => `${n < 0 ? "−" : ""}$${format.format(Math.abs(Math.round(n)))}`;
async function runSim() {
  if (simRunning) return;
  const bankroll = Number($("sim-bankroll").value), unit = Number($("sim-unit").value), shoes = Number($("sim-shoes").value);
  const rawSeed = $("sim-seed").value.trim();
  if (!Number.isFinite(bankroll) || bankroll < 25 || bankroll > 10000000) return toast("Bankroll must be between $25 and $10,000,000.");
  if (![5, 10, 25, 100].includes(unit)) return toast("Unit must be $5, $10, $25, or $100.");
  if (!Number.isInteger(shoes) || shoes < 1 || shoes > 2000) return toast("Shoes must be a whole number from 1 to 2,000.");
  const body = {bankroll_dollars: bankroll, unit_dollars: unit, max_shoes: shoes};
  if (rawSeed) { body.seed = Number(rawSeed); if (!Number.isSafeInteger(body.seed) || body.seed < 0 || body.seed >= 2**32) return toast("Use a whole seed between 0 and 4,294,967,295."); }
  simRunning = true; $("sim-run").disabled = true; $("sim-run").textContent = "Running…"; $("sim-outcome").textContent = "Running…";
  try {
    const sim = await api("/api/sim/run", body);
    const k = sim.kelly, f = sim.flat;
    $("sim-outcome").textContent = sim.outcome === "doubled" ? `Doubled in ${format.format(sim.rounds)} rounds` : sim.outcome === "ruined" ? `Busted after ${format.format(sim.rounds)} rounds` : `${format.format(sim.rounds)} rounds · no stop hit`;
    plot("sim-chart", [{color: "#277664", points: k.curve}, {color: "#c94732", points: f.curve}], {width: 800, height: 330, selectedX: null});
    const rows = [
      ["Kelly terminal", `${simDollars(k.terminal_dollars)} (${signed((k.terminal_dollars - sim.start_dollars) / sim.start_dollars * 100, 1)}%)`],
      ["Flat terminal", `${simDollars(f.terminal_dollars)} (${signed((f.terminal_dollars - sim.start_dollars) / sim.start_dollars * 100, 1)}%)`],
      ["Kelly peak / worst", `${simDollars(k.peak_dollars)} / ${(k.worst_fraction * 100).toFixed(0)}% of roll`],
      ["Bets / staked / max", `${format.format(k.bets)} / ${format.format(k.staked_units)}u / ${k.max_stake_units}u`],
      ["Sizing", sim.sizing],
    ];
    $("sim-stats").replaceChildren();
    for (const [label, value] of rows) {
      const div = document.createElement("div"), heading = document.createElement("p"), amount = document.createElement("strong");
      heading.textContent = label; amount.textContent = value; div.append(heading, amount); $("sim-stats").append(div);
    }
    $("sim-note").textContent = sim.note;
  } catch (error) { toast(error.message); $("sim-outcome").textContent = "Run failed"; }
  finally { simRunning = false; $("sim-run").disabled = false; $("sim-run").textContent = "Run it"; }
}
$("sim-run").addEventListener("click", runSim);
document.addEventListener("visibilitychange", () => { if (document.hidden && running) { stopping = true; controls(); } });
window.addEventListener("resize", () => { drawProfit(); if (training && !$("training-view").hidden) renderTraining(); });

(async function init() {  try {
    const status = await api("/api/status"); evidence = status.evidence;
    if (evidence) {
      $("evidence-value").textContent = `${pct(evidence.return_per_initial_unit_bet)} return per initial unit`;
      $("evidence-copy").textContent = `95% interval ${pct(evidence.return_per_initial_unit_bet_95ci[0])} to ${pct(evidence.return_per_initial_unit_bet_95ci[1])}. ${format.format(evidence.bets)} bets across ${format.format(evidence.observed_rounds)} observed rounds. Bet on ${(evidence.participation * 100).toFixed(2)}% of rounds.`;
    } else { $("evidence-value").textContent = "No saved evaluation"; $("evidence-copy").textContent = "Live session statistics appear above."; }
    const previous = storageGet("edge-session");
    if (previous) { try { render(await api(`/api/edge/session?id=${encodeURIComponent(previous)}`)); } catch (error) { if (!error.message.includes("Session expired")) throw error; } }
    if (!state) render(await api("/api/edge/new",{}));
    controls();
  } catch (error) { $("run-status").textContent = "Could not connect"; toast(`${error.message} Refresh to retry.`); }
})();
