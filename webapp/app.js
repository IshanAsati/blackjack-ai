/* Static SPA: all inference + simulation runs locally. Same frozen nets as the
   Python server (weights in data/*.json), same rules, own card streams. */
import {mulberry32, handValue, countTag, mlpForward, encodeClassic, encodeCount,
  Shoe, playRound, ClassicGame, drawUnlimited, kellyUnits} from "./engine.js";

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");
const sgn = (n, d = 1) => `${n < 0 ? "−" : "+"}${Math.abs(n).toFixed(d)}`;
const cardLabel = (v) => (v === 1 ? "A" : String(v));
let toastTimer;
function toast(msg) { $("toast").textContent = msg; $("toast").classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => $("toast").classList.remove("show"), 5000); }

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Missing ${path} — run: python -m blackjackai export`);
  return r.json();
}
const [classicQ, countQ, betValue, pairsData, fixedPlay, sizingData, training] = await Promise.all([
  getJSON("data/classic_q.json"), getJSON("data/count_q.json"), getJSON("data/bet_value.json"),
  getJSON("data/pairs.json"), getJSON("data/fixed_play.json"), getJSON("data/sizing.json"),
  getJSON("data/training.json"),
]).catch((e) => { document.body.innerHTML = `<p style="padding:40px">${e.message}</p>`; throw e; });

const RULES = {decks: 6, penetration: 0.75, blackjack_payout: 1.5, double_after_split: true, ...(betValue.rules || {})};
const pairSet = new Set(pairsData.splits.map(([r, u]) => `${r},${u}`));
const SIZING = {kelly: sizingData.selected.kelly_fraction, cap: sizingData.selected.cap_units, minEdge: sizingData.selected.min_edge};

/* ---------- shared helpers ---------- */
function cardHTML(v, hidden = false) {
  if (hidden) return `<div class="card hidden" aria-label="Hidden card"></div>`;
  return `<div class="card" aria-label="${cardLabel(v)}"><span>${cardLabel(v)}</span><span>${cardLabel(v)}</span></div>`;
}
function drawChart(svgId, series, counts) {
  const svg = $(svgId), W = 640, H = 260, pad = {l: 58, r: 12, t: 14, b: 26};
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) { svg.innerHTML = ""; return; }
  const xmax = Math.max(1, ...pts.map((p) => p[0]));
  let ymin = Math.min(0, ...pts.map((p) => p[1])), ymax = Math.max(0, ...pts.map((p) => p[1]));
  const m = Math.max((ymax - ymin) * 0.12, 1); ymin -= m; ymax += m;
  const X = (v) => pad.l + (v / xmax) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - ((v - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
  let s = "";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + ((ymax - ymin) * i) / 4, y = Y(v), x = X((xmax * i) / 4);
    s += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y}" y2="${y}" stroke="#ddd5c4"/><text x="${pad.l - 6}" y="${y + 4}" text-anchor="end" font-size="11" fill="#68777b">${counts ? (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%" : fmt.format(Math.round(v))}</text>`;
    s += `<text x="${x}" y="${H - 8}" text-anchor="middle" font-size="11" fill="#68777b">${fmt.format(Math.round((xmax * i) / 4))}</text>`;
  }
  const colors = ["#277664", "#c94732"];
  series.forEach((line, i) => { s += `<polyline points="${line.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(" ")}" fill="none" stroke="${colors[i % 2]}" stroke-width="2.5"/>`; });
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = s;
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => { x.classList.toggle("selected", x === b); x.setAttribute("aria-pressed", x === b); });
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `tab-${b.dataset.tab}`; });
}));

/* ---------- count table (home) ---------- */
function countDecide(total, up, ace, tc, frac, first) {
  const q = mlpForward(countQ.network, encodeCount(total, up, ace, tc, frac, first));
  if (!first) q[2] = -Infinity;
  return q.indexOf(Math.max(...q));
}
const table = {rng: mulberry32((Math.random() * 2 ** 32) >>> 0), shoe: null, shoeN: 0,
  rounds: 0, bets: 0, skips: 0, profit: 0, baseline: 0, roll: 400, sized: 0, sizedBase: 0,
  history: [[0, 0]], auto: false, last: null};
function newShoe() { table.shoe = new Shoe(RULES, table.rng); table.shoeN++; }
function tableRound() {
  if (!table.shoe || !table.shoe.canDeal) newShoe();
  const shoe = table.shoe;
  const {trueCount: tc, fraction} = shoe.observation();
  const edge = mlpForward(betValue.network, [Math.max(-15, Math.min(15, tc)) / 10, fraction])[0];
  const stake = kellyUnits(edge, table.roll, {kelly: SIZING.kelly, cap: SIZING.cap, minEdge: SIZING.minEdge});
  const bet = stake > 0;
  const drawn = [];
  const draw = () => { const c = shoe.cards.pop(); drawn.push(c); return c; };
  const decide = (total, up, ace, first) => countDecide(total, up, ace, tc, fraction, first);
  const fixed = {get: (k) => { const [t, u, a, f] = k.split(",").map(Number); return decide(t, u, a, f); }};
  const round = playRound(shoe, draw, fixed, pairSet, true);
  shoe.exposed += drawn.reduce((a, c) => a + countTag(c), 0);
  const profit = bet ? stake * round.reward : 0;
  table.rounds++;
  if (bet) { table.bets++; table.staked = (table.staked || 0) + stake; } else table.skips++;
  table.profit += bet ? round.reward : 0;
  table.baseline += round.reward;
  table.sized += profit; table.sizedBase += stake * round.reward;
  table.roll += profit;
  table.last = {...round, bet, stake, edge, tc, fraction, shoeN: table.shoeN};
  if (table.rounds % 100 === 0) table.history.push([table.rounds, table.sized]);
  if (table.history.length > 400) table.history = table.history.filter((_, i) => i % 2 === 0);
  return table.last;
}
function renderTable() {
  const t = table;
  $("t-profit").innerHTML = `${sgn(t.sized)} <small>units</small>`;
  $("t-profit").style.color = t.sized < 0 ? "#c94732" : "#17313a";
  $("t-rounds").textContent = fmt.format(t.rounds);
  $("t-bets").textContent = fmt.format(t.bets);
  $("t-skips").textContent = fmt.format(t.skips);
  const shoe = t.shoe;
  if (shoe) {
    const {trueCount, fraction} = shoe.observation();
    $("t-shoe").textContent = `Shoe ${t.shoeN}`;
    $("t-left").textContent = `${shoe.cards.length} cards left`;
    $("t-fill").style.width = `${fraction * 100}%`;
    $("t-tc").textContent = sgn(trueCount, 2);
  }
  const last = t.last;
  if (last) {
    $("t-decision").textContent = last.bet ? (last.stake > 1 ? `Bet ${last.stake} units.` : "Bet one unit.") : "Sit this one out.";
    $("t-decision-sub").textContent = last.bet ? `Committed before the deal at count ${sgn(last.tc, 2)}.` : "Waiting while the table keeps playing.";
    $("t-ev").textContent = `${sgn(last.edge * 100, 2)}%`;
    $("t-stake").textContent = last.bet ? `${last.stake}u` : "—";
    $("t-heading").textContent = `Round ${fmt.format(last.round || t.rounds)} · complete`;
    $("t-lastbet").textContent = last.bet ? `${sgn(last.stake * last.reward)} · ${last.stake}u placed` : "Observed · no wager";
    $("t-dealer").innerHTML = last.dealer.map((c) => cardHTML(c)).join("");
    $("t-dtotal").textContent = last.dealer_total;
    $("t-plabel").textContent = last.bet ? "Our hand" : "Table";
    if (last.split) {
      $("t-player").innerHTML = last.hands.map((h, i) => `<div><small style="color:#b8cdca">H${i + 1} · ${h.total}</small><div style="display:flex">${h.cards.map((c) => cardHTML(c)).join("")}</div></div>`).join("");
      $("t-ptotal").textContent = last.hands.map((h) => h.total).join(" / ");
    } else {
      $("t-player").innerHTML = last.player.map((c) => cardHTML(c)).join("");
      $("t-ptotal").textContent = last.player_total;
    }
    const mv = last.split ? last.hands.map((h, i) => `H${i + 1} ${h.moves.join("→")}`).join(" · ") : last.moves.join(" → ");
    $("t-note").textContent = `${mv || "Blackjack"}. Count ${sgn(last.tc, 2)}.`;
  }
  $("t-roi").textContent = table.staked ? `${sgn((t.sized / table.staked) * 100, 2)}% / unit staked` : "—";
  $("t-roll").textContent = `${t.roll.toFixed(1)} u`;
  drawChart("t-chart", [{points: t.history}], false);
  $("t-auto").textContent = t.auto ? "Pause auto" : "Start auto";
}
$("t-next").addEventListener("click", () => { tableRound(); renderTable(); });
$("t-find").addEventListener("click", () => {
  for (let i = 0; i < 10000; i++) { const r = tableRound(); if (r.bet) break; }
  renderTable();
});
$("t-new").addEventListener("click", () => {
  Object.assign(table, {rng: mulberry32((Math.random() * 2 ** 32) >>> 0), shoe: null, shoeN: 0,
    rounds: 0, bets: 0, skips: 0, staked: 0, profit: 0, baseline: 0, roll: 400, sized: 0, sizedBase: 0, history: [[0, 0]], last: null});
  renderTable();
});
$("t-auto").addEventListener("click", () => {
  table.auto = !table.auto;
  renderTable();
  const tick = () => {
    if (!table.auto) return;
    for (let i = 0; i < 200; i++) tableRound();
    renderTable();
    setTimeout(tick, 60);
  };
  if (table.auto) tick();
});
renderTable();

/* ---------- classic solo play ---------- */
const solo = {rng: Math.random, game: null, score: JSON.parse(localStorage.getItem("bj-score") || '{"win":0,"draw":0,"loss":0}')};
function soloAdvice(obs) {
  const q = mlpForward(classicQ.network, encodeClassic(obs.total, obs.dealer, obs.ace));
  return {action: q[1] > q[0] ? "hit" : "stand", qs: q[0], qh: q[1], margin: Math.abs(q[1] - q[0])};
}
function renderSolo() {
  const g = solo.game;
  if (!g) return;
  $("p-player").innerHTML = g.player.map((c) => cardHTML(c)).join("");
  $("p-ptotal").textContent = handValue(g.player).total;
  $("p-dealer").innerHTML = g.over
    ? g.dealer.map((c) => cardHTML(c)).join("")
    : cardHTML(g.dealer[0]) + cardHTML(null, true);
  $("p-dtotal").textContent = g.over ? handValue(g.dealer).total : `${g.dealer[0]}+?`;
  const playing = !g.over;
  $("p-hit").disabled = $("p-stand").disabled = !playing;
  if (playing) {
    const a = soloAdvice(g.observation());
    $("p-advice").textContent = a.action === "hit" ? "Hit" : "Stand";
    $("p-advice-note").textContent = `Prefers ${a.action} by ${a.margin.toFixed(3)} value points.`;
    $("p-vs").textContent = sgn(a.qs, 3); $("p-vh").textContent = sgn(a.qh, 3);
    $("p-note").textContent = "Choose your move.";
  } else {
    $("p-advice").textContent = g.result === "win" ? "You won" : g.result === "loss" ? "Dealer wins" : "Push";
    $("p-advice-note").textContent = `Final: you ${handValue(g.player).total}, dealer ${handValue(g.dealer).total}.`;
    solo.score[g.result]++;
    localStorage.setItem("bj-score", JSON.stringify(solo.score));
  }
  $("p-score").textContent = `${solo.score.win} / ${solo.score.draw} / ${solo.score.loss}`;
}
$("p-deal").addEventListener("click", () => { solo.game = new ClassicGame(Math.random); renderSolo(); });
$("p-hit").addEventListener("click", () => { solo.game.hit(); renderSolo(); });
$("p-stand").addEventListener("click", () => { solo.game.stand(); renderSolo(); });
document.addEventListener("keydown", (e) => {
  if ($("tab-play").hidden || !solo.game || solo.game.over) return;
  if (e.key.toLowerCase() === "h") { solo.game.hit(); renderSolo(); }
  if (e.key.toLowerCase() === "s") { solo.game.stand(); renderSolo(); }
});
$("p-score").textContent = `${solo.score.win} / ${solo.score.draw} / ${solo.score.loss}`;

/* ---------- shared multiplayer table ---------- */
const multi = {rng: Math.random, seats: [], dealer: [], current: null, over: true};
function multiDeal() {
  const n = Number($("m-seats").value);
  multi.seats = Array.from({length: n}, () => ({cards: [drawUnlimited(multi.rng), drawUnlimited(multi.rng)], status: "playing", result: null}));
  multi.dealer = [drawUnlimited(multi.rng), drawUnlimited(multi.rng)];
  multi.current = 0; multi.over = false;
  renderMulti();
}
function multiAct(seat, action) {
  const s = multi.seats[seat];
  if (action === "hit") {
    s.cards.push(drawUnlimited(multi.rng));
    if (handValue(s.cards).total > 21) { s.status = "bust"; s.result = "loss"; }
  } else s.status = "stood";
  multi.current = multi.seats.findIndex((x) => x.status === "playing");
  if (multi.current === -1) {
    multi.current = null; multi.over = true;
    const live = multi.seats.filter((x) => x.status === "stood");
    if (live.length) while (handValue(multi.dealer).total < 17) multi.dealer.push(drawUnlimited(multi.rng));
    const d = handValue(multi.dealer).total;
    for (const x of live) {
      const p = handValue(x.cards).total;
      x.result = d > 21 || p > d ? "win" : p < d ? "loss" : "draw";
    }
    multi.seats.forEach((x) => { if (x.status !== "playing") x.status = "finished"; });
  }
  renderMulti();
}
function renderMulti() {
  $("m-dealer").innerHTML = multi.over || !multi.dealer.length ? multi.dealer.map((c) => cardHTML(c)).join("")
    : cardHTML(multi.dealer[0]) + cardHTML(null, true);
  $("m-dtotal").textContent = !multi.dealer.length ? "—" : multi.over ? handValue(multi.dealer).total : `${multi.dealer[0]}+?`;
  $("m-turn").textContent = multi.over ? (multi.seats.length ? "hand complete" : "deal to begin") : `seat ${multi.current + 1} to act`;
  const box = $("m-seats"); box.innerHTML = "";
  $("m-advice").innerHTML = "";
  multi.seats.forEach((s, i) => {
    const t = handValue(s.cards).total;
    const div = document.createElement("div");
    div.className = `seat${i === multi.current ? " active" : ""}`;
    let tip = s.result || s.status;
    if (s.status === "playing") {
      const q = mlpForward(classicQ.network, encodeClassic(t, multi.dealer[0], handValue(s.cards).soft ? 1 : 0));
      tip = `model: ${q[1] > q[0] ? "hit" : "stand"}`;
      const row = document.createElement("div"); row.innerHTML = `<span>Seat ${i + 1} · ${t}</span><span class="tip">${tip}</span>`;
      $("m-advice").append(row);
    }
    div.innerHTML = `<h3>Seat ${i + 1} · ${t}${s.result ? ` · ${s.result}` : ""}</h3><div class="cards">${s.cards.map((c) => cardHTML(c)).join("")}</div><div class="tip">${i === multi.current ? "To act" : tip}</div>`;
    box.append(div);
  });
  $("m-hit").disabled = $("m-stand").disabled = $("m-ai").disabled = multi.over;
}
$("m-deal").addEventListener("click", multiDeal);
$("m-hit").addEventListener("click", () => multiAct(multi.current, "hit"));
$("m-stand").addEventListener("click", () => multiAct(multi.current, "stand"));
$("m-ai").addEventListener("click", () => {
  const s = multi.seats[multi.current], t = handValue(s.cards).total;
  const q = mlpForward(classicQ.network, encodeClassic(t, multi.dealer[0], handValue(s.cards).soft ? 1 : 0));
  multiAct(multi.current, q[1] > q[0] ? "hit" : "stand");
});

/* ---------- advisor ---------- */
const RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const adv = {dealer: null, hand: []};
for (const v of RANKS) {
  const b = document.createElement("button");
  b.textContent = v === 1 ? "A" : v;
  b.addEventListener("click", () => { adv.dealer = v; renderAdv(); });
  $("a-dealer").append(b);
  const c = document.createElement("button");
  c.textContent = `+${v === 1 ? "A" : v}`;
  c.addEventListener("click", () => { adv.hand.push(v); renderAdv(); });
  $("a-player").append(c);
}
function renderAdv() {
  [...$("a-dealer").children].forEach((b, i) => b.classList.toggle("sel", RANKS[i] === adv.dealer));
  $("a-hand").innerHTML = adv.hand.length ? adv.hand.map((c) => cardHTML(c)).join("") : "<span style='color:#8fb0ad'>Tap + cards to build the hand</span>";
  if (adv.dealer === null || adv.hand.length < 2) {
    $("a-verdict").textContent = "Pick the cards";
    $("a-note").textContent = "Dealer card, then at least two cards for your hand.";
    $("a-total").textContent = adv.hand.length ? handValue(adv.hand).total : "";
    return;
  }
  const {total, soft} = handValue(adv.hand);
  $("a-total").textContent = total > 21 ? `${total} (bust)` : soft ? `soft ${total}` : total;
  if (total > 21) { $("a-verdict").textContent = "Bust"; $("a-note").textContent = "Over 21 — no decision left."; return; }
  if (total === 21) { $("a-verdict").textContent = "Stand"; $("a-note").textContent = "21 — hitting always busts."; return; }
  const q = mlpForward(classicQ.network, encodeClassic(total, adv.dealer, soft ? 1 : 0));
  const action = q[1] > q[0] ? "hit" : "stand";
  $("a-verdict").textContent = action === "hit" ? "Hit" : "Stand";
  $("a-note").textContent = `${total}${soft ? " soft" : ""} vs dealer ${adv.dealer === 1 ? "A" : adv.dealer} — prefers ${action} by ${Math.abs(q[1] - q[0]).toFixed(3)}.`;
  $("a-vs").textContent = sgn(q[0], 3); $("a-vh").textContent = sgn(q[1], 3);
}
$("a-undo").addEventListener("click", () => { adv.hand.pop(); renderAdv(); });
$("a-clear").addEventListener("click", () => { adv.hand = []; renderAdv(); });
renderAdv();

/* ---------- bankroll sim ---------- */
$("s-run").addEventListener("click", () => {
  const dollars = Number($("s-roll").value), unit = Number($("s-unit").value), shoesN = Number($("s-shoes").value);
  const seedRaw = $("s-seed").value.trim();
  if (!(dollars >= 25 && dollars <= 10000000)) return toast("Bankroll must be $25–$10M.");
  if (![5, 10, 25, 100].includes(unit) || !(shoesN >= 1 && shoesN <= 1000)) return toast("Check unit and shoe count.");
  const seed = seedRaw ? Number(seedRaw) : (Math.random() * 2 ** 32) >>> 0;
  $("s-run").disabled = true; $("s-run").textContent = "Running…";
  setTimeout(() => {
    try {
      const startU = dollars / unit;
      const cap = Math.max(1, Math.round(SIZING.cap * (startU / 400)));
      const rng = mulberry32(seed);
      let roll = startU, flat = startU, rounds = 0, bets = 0, maxStake = 0, shoeN = 0;
      let shoe = null;
      const kCurve = [[0, dollars]], fCurve = [[0, dollars]];
      let outcome = "capped";
      const nextShoe = () => { shoe = new Shoe(RULES, rng); shoeN++; };
      nextShoe();
      while (shoeN <= shoesN && outcome === "capped") {
        if (!shoe.canDeal) { nextShoe(); continue; }
        const {trueCount: tc, fraction} = shoe.observation();
        const edge = mlpForward(betValue.network, [Math.max(-15, Math.min(15, tc)) / 10, fraction])[0];
        const stake = kellyUnits(edge, roll, {kelly: SIZING.kelly, cap, minEdge: SIZING.minEdge});
        const drawn = [];
        const draw = () => { const c = shoe.cards.pop(); drawn.push(c); return c; };
        const decide = (t, u, a, f) => countDecide(t, u, a, tc, fraction, f);
        const fixed = {get: (k) => { const [t, u, a, f] = k.split(",").map(Number); return decide(t, u, a, f); }};
        const r = playRound(shoe, draw, fixed, pairSet, false);
        shoe.exposed += drawn.reduce((a, c) => a + countTag(c), 0);
        if (stake > 0) { bets++; maxStake = Math.max(maxStake, stake); }
        roll += stake * r.reward;
        if (edge > betValue.threshold) flat += r.reward;
        rounds++;
        if (rounds % 25 === 0) { kCurve.push([rounds, +(roll * unit).toFixed(2)]); fCurve.push([rounds, +(flat * unit).toFixed(2)]); }
        if (roll <= 0) outcome = "ruined";
        else if (roll >= 2 * startU) outcome = "doubled";
      }
      kCurve.push([rounds, +(Math.max(roll, 0) * unit).toFixed(2)]);
      fCurve.push([rounds, +(Math.max(flat, 0) * unit).toFixed(2)]);
      $("s-outcome").textContent = outcome === "doubled" ? `Doubled in ${fmt.format(rounds)} rounds` : outcome === "ruined" ? `Busted after ${fmt.format(rounds)} rounds` : `${fmt.format(rounds)} rounds · no stop hit`;
      drawChart("s-chart", [{points: kCurve}, {points: fCurve}], false);
      const d = (u) => `$${fmt.format(Math.round(u * unit))}`;
      $("s-stats").innerHTML = [
        ["Kelly terminal", `${d(Math.max(roll, 0))}`], ["Flat terminal", `${d(Math.max(flat, 0))}`],
        ["Bets / max stake", `${fmt.format(bets)} / ${maxStake}u`], ["Seed", seed],
      ].map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join("");
    } catch (e) { toast(e.message); }
    $("s-run").disabled = false; $("s-run").textContent = "Run it";
  }, 30);
});

/* ---------- training tab ---------- */
{
  const pts = (run) => run.points.map((p) => [Number(p.episode), Number(p.exact_mean_reward) * 100]).filter((p) => Number.isFinite(p[1]));
  const series = training.runs.map((r) => ({points: pts(r)}));
  const svg = $("r-chart"), W = 780, H = 300, pad = {l: 62, r: 14, t: 14, b: 28};
  const OPT = -0.0465559770666975; // exact optimal, original unlimited-deck rules
  const all = series.flatMap((s) => s.points);
  const xmax = Math.max(1, ...all.map((p) => p[0]));
  let ymin = Math.min(OPT * 100, ...all.map((p) => p[1])), ymax = Math.max(OPT * 100, ...all.map((p) => p[1]));
  const m = (ymax - ymin) * 0.15; ymin -= m; ymax += m;
  const X = (v) => pad.l + (v / xmax) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - ((v - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
  let s = `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(OPT * 100)}" y2="${Y(OPT * 100)}" stroke="#8a743b" stroke-dasharray="6 4"/>`;
  const colors = ["#277664", "#c94732"];
  series.forEach((line, i) => { s += `<polyline points="${line.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(" ")}" fill="none" stroke="${colors[i % 2]}" stroke-width="2.5"/>`; });
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = s;
  const cards = [];
  if (training.distilled) cards.push(["Count-aware strategist", `${(100 * training.distilled.grid_agreement).toFixed(1)}% analytical agreement`]);
  if (training.betting_validation) cards.push(["Betting validation", `threshold ${(100 * training.betting_validation.selected_threshold).toFixed(1)}%`]);
  cards.push(["Optimal (exact)", "−4.66% / hand"]);
  $("r-cards").innerHTML = cards.map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join("");
}
