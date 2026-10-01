const $ = (id) => document.getElementById(id);
const RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const label = (v) => (v === 1 ? "A" : String(v));

let dealer = null;
let hand = [];

function buildPickers() {
  for (const v of RANKS) {
    const b = document.createElement("button");
    b.textContent = label(v);
    b.setAttribute("aria-label", v === 1 ? "Dealer ace" : `Dealer ${v}`);
    b.addEventListener("click", () => { dealer = v; renderPickers(); advise(); });
    $("dealer-picker").append(b);
  }
  for (const v of RANKS) {
    const b = document.createElement("button");
    b.textContent = `+${label(v)}`;
    b.setAttribute("aria-label", v === 1 ? "Add ace" : `Add ${v}`);
    b.addEventListener("click", () => { hand.push(v); renderPickers(); advise(); });
    $("player-picker").append(b);
  }
}

function renderPickers() {
  [...$("dealer-picker").children].forEach((b, i) => b.classList.toggle("selected", RANKS[i] === dealer));
  $("undo").disabled = $("clear").disabled = hand.length === 0;
  const box = $("hand-cards");
  box.innerHTML = "";
  if (!hand.length) {
    const hint = document.createElement("span");
    hint.className = "empty-hand";
    hint.textContent = "Tap cards below to add them";
    box.append(hint);
    $("hand-total").textContent = "—";
    return;
  }
  hand.forEach((v, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.title = "Remove this card";
    card.setAttribute("aria-label", `Remove ${label(v)}`);
    card.innerHTML = `<span>${label(v)}</span><span>${label(v)}</span>`;
    card.addEventListener("click", () => { hand.splice(i, 1); renderPickers(); advise(); });
    box.append(card);
  });
}

function signed(value) { return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`; }

function showWaiting(copy) {
  $("status-label").textContent = "Enter the cards";
  $("result-copy").textContent = copy || "Pick the dealer's card, then build your hand. The model answers from the same position.";
  $("advice-context").textContent = "Waiting for cards";
  $("advice-action").textContent = "—";
  $("advice-action").classList.remove("bust");
  $("advice-note").textContent = "Choose a dealer card and at least two cards for your hand.";
  $("values").hidden = true;
}

let pending = null;
async function advise() {
  renderHandTotal();
  if (dealer === null || hand.length < 2) {
    showWaiting(dealer === null ? "Start by picking the dealer's visible card." : hand.length ? "Add at least one more card to your hand." : undefined);
    return;
  }
  const body = {player: hand, dealer};
  const request = (async () => {
    const response = await fetch("/api/advise", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Advice failed.");
    return data;
  })();
  pending = request;
  try {
    const state = await request;
    if (pending !== request) return; // a newer request superseded this one
    renderAdvice(state);
  } catch (error) {
    if (pending !== request) return;
    showToast(error.message);
  }
}

function handTotal(cards) {
  let total = cards.reduce((a, b) => a + (b === 1 ? 1 : b), 0);
  let aces = cards.filter((c) => c === 1).length;
  let soft = false;
  while (aces > 0 && total + 10 <= 21) { total += 10; aces -= 1; soft = true; }
  return {total, soft};
}

function renderHandTotal() {
  if (!hand.length) { $("hand-total").textContent = "—"; return; }
  const {total, soft} = handTotal(hand);
  $("hand-total").textContent = total > 21 ? `${total} (bust)` : soft ? `soft ${total}` : String(total);
}

function renderAdvice(state) {
  if (state.status === "bust") {
    $("status-label").textContent = "That's a bust";
    $("result-copy").textContent = `${state.player_total} is over 21 — no decision left. Undo a card to try again.`;
    $("advice-context").textContent = `You ${state.player_total} vs dealer ${label(state.dealer)}`;
    $("advice-action").textContent = "Bust";
    $("advice-action").classList.add("bust");
    $("advice-note").textContent = state.note;
    $("values").hidden = true;
    return;
  }
  $("advice-action").classList.remove("bust");
  $("status-label").textContent = state.status === "blackjack" ? "Blackjack" : "Your decision";
  $("result-copy").textContent = state.status === "blackjack"
    ? "21 — stand and collect."
    : `You have ${state.player_total}${state.usable_ace ? " (soft, ace counting as 11)" : ""} against dealer ${label(state.dealer)}.`;
  $("advice-context").textContent = `${state.player_total}${state.usable_ace ? " soft" : ""} vs dealer ${label(state.dealer)}`;
  $("advice-action").textContent = state.advice.action === "hit" ? "Hit" : "Stand";
  $("advice-note").textContent = (state.note ? state.note + " " : "") +
    `The network prefers ${state.advice.action} by ${state.advice.margin.toFixed(3)} value points.`;
  $("stand-value").textContent = signed(state.advice.q_stand);
  $("hit-value").textContent = signed(state.advice.q_hit);
  $("values").hidden = false;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 3500);
}

$("undo").addEventListener("click", () => { hand.pop(); renderPickers(); advise(); });
$("clear").addEventListener("click", () => { hand = []; renderPickers(); advise(); });
document.addEventListener("keydown", (event) => {
  if (event.key === "Backspace" && hand.length) { hand.pop(); renderPickers(); advise(); return; }
  const key = event.key.toLowerCase();
  const rank = key === "a" ? 1 : key === "t" || key === "0" ? 10 : Number.parseInt(key, 10);
  if (RANKS.includes(rank)) { hand.push(rank); renderPickers(); advise(); }
});

buildPickers();
renderPickers();
showWaiting();
