const $ = (id) => document.getElementById(id);
const ui = {
  deal: $("deal"), hit: $("hit"), stand: $("stand"),
  playerCards: $("player-cards"), dealerCards: $("dealer-cards"),
  playerTotal: $("player-total"), dealerTotal: $("dealer-total"),
  status: $("status-label"), copy: $("result-copy"), advice: $("advice-action"),
  adviceNote: $("advice-note"), values: $("values"), standValue: $("stand-value"),
  hitValue: $("hit-value"), agreement: $("agreement"), agreementSymbol: $("agreement-symbol"),
  agreementTitle: $("agreement-title"), agreementCopy: $("agreement-copy"), toast: $("toast"),
};

let game = null;
let recommendation = null;
let score = JSON.parse(localStorage.getItem("blackjack-score") || '{"win":0,"draw":0,"loss":0}');
let scoredGame = null;

function cardMarkup(value, index) {
  if (value === null) return '<div class="card hidden" aria-label="Hidden card"></div>';
  const label = value === 1 ? "A" : value === 10 ? ["10", "J", "Q", "K"][index % 4] : String(value);
  const suit = ["♠", "♥", "♣", "♦"][index % 4];
  const red = suit === "♥" || suit === "♦" ? " red" : "";
  return `<div class="card${red}" aria-label="${label} of ${suit}"><span>${label}${suit}</span><span>${label}${suit}</span></div>`;
}

function renderCards(container, cards) {
  container.innerHTML = cards.map(cardMarkup).join("");
}

function renderScore() {
  $("wins").textContent = score.win;
  $("draws").textContent = score.draw;
  $("losses").textContent = score.loss;
  localStorage.setItem("blackjack-score", JSON.stringify(score));
}

function render(state) {
  game = state;
  renderCards(ui.playerCards, state.player);
  renderCards(ui.dealerCards, state.dealer);
  ui.playerTotal.textContent = state.player_total;
  ui.dealerTotal.textContent = state.status === "playing" ? `${state.dealer_total}+?` : state.dealer_total;
  const playing = state.status === "playing";
  ui.hit.disabled = !playing;
  ui.stand.disabled = !playing;
  ui.deal.textContent = playing ? "New hand" : "Deal again";

  if (playing) {
    recommendation = state.advice.action;
    ui.status.textContent = "Your decision";
    ui.copy.textContent = "Choose your move, then see whether the network read it the same way.";
    ui.advice.textContent = recommendation;
    ui.adviceNote.textContent = `The network prefers ${recommendation} by ${state.advice.margin.toFixed(3)} value points.`;
    ui.standValue.textContent = signed(state.advice.q_stand);
    ui.hitValue.textContent = signed(state.advice.q_hit);
    ui.values.hidden = false;
  } else {
    recommendation = null;
    ui.status.textContent = state.result === "win" ? "You won" : state.result === "loss" ? "Dealer wins" : "Push";
    ui.copy.textContent = resultCopy(state);
    ui.advice.textContent = state.result === "win" ? "Well played" : state.result === "draw" ? "Even hand" : "Next hand";
    ui.adviceNote.textContent = `Final totals: you ${state.player_total}, dealer ${state.dealer_total}.`;
    ui.values.hidden = true;
    if (scoredGame !== state.id) {
      score[state.result] += 1;
      scoredGame = state.id;
      renderScore();
    }
  }
}

function signed(value) { return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`; }

function resultCopy(state) {
  if (state.player_total > 21) return `Your hand went over 21. The dealer takes this one.`;
  if (state.dealer_total > 21) return `The dealer went over 21. Your hand holds.`;
  if (state.result === "win") return `Your ${state.player_total} beats the dealer’s ${state.dealer_total}.`;
  if (state.result === "loss") return `The dealer’s ${state.dealer_total} beats your ${state.player_total}.`;
  return `Both hands finish at ${state.player_total}. No win, no loss.`;
}

function showAgreement(action, advised) {
  const agreed = action === advised;
  ui.agreement.hidden = false;
  ui.agreement.classList.toggle("disagreed", !agreed);
  ui.agreementSymbol.textContent = agreed ? "✓" : "↗";
  ui.agreementTitle.textContent = agreed ? "You agreed" : "You chose differently";
  ui.agreementCopy.textContent = agreed ? `Both you and the model chose ${action}.` : `You chose ${action}; the model preferred ${advised}.`;
}

async function request(path, body = {}) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The table could not complete that move.");
  return data;
}

async function deal() {
  setBusy(true);
  try {
    ui.agreement.hidden = true;
    render(await request("/api/game/new"));
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function act(action) {
  if (!game || game.status !== "playing") return;
  const advised = recommendation;
  setBusy(true);
  try {
    const state = await request(`/api/game/${action}`, { id: game.id });
    showAgreement(action, advised);
    render(state);
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

function setBusy(busy) {
  ui.deal.disabled = busy;
  ui.hit.disabled = busy || !game || game.status !== "playing";
  ui.stand.disabled = busy || !game || game.status !== "playing";
}

function showError(error) {
  ui.toast.textContent = error.message;
  ui.toast.classList.add("visible");
  setTimeout(() => ui.toast.classList.remove("visible"), 3500);
}

ui.deal.addEventListener("click", deal);
ui.hit.addEventListener("click", () => act("hit"));
ui.stand.addEventListener("click", () => act("stand"));
$("reset-score").addEventListener("click", () => { score = { win: 0, draw: 0, loss: 0 }; renderScore(); });
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "h" && !ui.hit.disabled) act("hit");
  if (event.key.toLowerCase() === "s" && !ui.stand.disabled) act("stand");
});

renderScore();
