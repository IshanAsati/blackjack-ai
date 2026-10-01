const $ = (id) => document.getElementById(id);
const ui = {
  deal: $("deal"), hit: $("hit"), stand: $("stand"), ai: $("ai-move"),
  seats: $("seats"), seatList: $("seat-list"), seatCount: $("seat-count"),
  dealerCards: $("dealer-cards"), dealerTotal: $("dealer-total"),
  status: $("status-label"), copy: $("result-copy"),
  adviceSeat: $("advice-seat"), advice: $("advice-action"), adviceNote: $("advice-note"),
  values: $("values"), standValue: $("stand-value"), hitValue: $("hit-value"),
  toast: $("toast"),
};

let table = null;
let selectedSeat = 0;
let score = JSON.parse(localStorage.getItem("multi-score") || '{"win":0,"draw":0,"loss":0}');
let scoredTable = null;

function cardMarkup(value, index) {
  if (value === null) return '<div class="card hidden" aria-label="Hidden card"></div>';
  const label = value === 1 ? "A" : value === 10 ? ["10", "J", "Q", "K"][index % 4] : String(value);
  const suit = ["♠", "♥", "♣", "♦"][index % 4];
  const red = suit === "♥" || suit === "♦" ? " red" : "";
  return `<div class="card${red}"><span>${label}${suit}</span><span>${label}${suit}</span></div>`;
}

function renderScore() {
  $("wins").textContent = score.win;
  $("draws").textContent = score.draw;
  $("losses").textContent = score.loss;
  localStorage.setItem("multi-score", JSON.stringify(score));
}

function seatAdvice(seat) {
  return table && table.advice ? table.advice[String(seat)] : null;
}

function render(state) {
  table = state;
  if (state.current_seat !== null && state.current_seat !== undefined) selectedSeat = state.current_seat;
  selectedSeat = Math.max(0, Math.min(selectedSeat, state.seats.length - 1));
  ui.dealerCards.innerHTML = state.dealer.map(cardMarkup).join("");
  ui.dealerTotal.textContent = state.stage === "playing" ? `${state.dealer_total}+?` : state.dealer_total;

  ui.seats.innerHTML = "";
  ui.seatList.innerHTML = "";
  for (const seat of state.seats) {
    const advice = seatAdvice(seat.seat);
    const card = document.createElement("div");
    card.className = `seat${seat.seat === state.current_seat ? " active" : ""}${seat.status === "finished" ? " done" : ""}`;
    const result = seat.result ? ` · ${seat.result}` : seat.status === "bust" ? " · bust" : seat.status === "stood" ? " · stood" : "";
    const tip = advice ? `model: ${advice.action}` : seat.result || seat.status;
    card.innerHTML = `<div class="seat-head"><strong>Seat ${seat.seat + 1}</strong><span class="total">${seat.total}</span></div>
      <div class="cards">${seat.cards.map(cardMarkup).join("")}</div>
      <p class="seat-meta">${seat.seat === state.current_seat ? "To act" : tip}${result}</p>`;
    const pick = document.createElement("button");
    pick.textContent = `Play seat ${seat.seat + 1}`;
    pick.disabled = state.stage !== "playing" || seat.status !== "playing";
    pick.addEventListener("click", () => { selectedSeat = seat.seat; render(table); });
    card.appendChild(pick);
    ui.seats.appendChild(card);

    const row = document.createElement("div");
    row.className = `seat-row${seat.seat === selectedSeat ? " active" : ""}`;
    row.innerHTML = `<span>Seat ${seat.seat + 1} · ${seat.total}${result}</span><span class="tip">${tip}</span>`;
    row.addEventListener("click", () => { selectedSeat = seat.seat; render(table); });
    ui.seatList.appendChild(row);
  }

  const playing = state.stage === "playing";
  const current = state.current_seat;
  ui.hit.disabled = !playing;
  ui.stand.disabled = !playing;
  ui.ai.disabled = !playing;
  ui.deal.textContent = playing ? "Restart table" : "Deal again";

  if (playing) {
    const advice = table.current_advice;
    ui.status.textContent = `Seat ${current + 1} to act`;
    ui.copy.textContent = "Seats act in order. The rail shows the current seat; the list shows every live seat.";
    ui.adviceSeat.textContent = `Seat ${current + 1} · model advice`;
    ui.advice.textContent = advice.action;
    ui.adviceNote.textContent = `Prefers ${advice.action} by ${advice.margin.toFixed(3)} value points.`;
    ui.standValue.textContent = signed(advice.q_stand);
    ui.hitValue.textContent = signed(advice.q_hit);
    ui.values.hidden = false;
  } else {
    ui.status.textContent = "Hand complete";
    const wins = state.seats.filter((s) => s.result === "win").length;
    ui.copy.textContent = wins ? `${wins} of ${state.seats.length} seats beat the dealer.` : "The dealer holds this time.";
    ui.adviceSeat.textContent = "Hand complete";
    ui.advice.textContent = "Settled";
    ui.adviceNote.textContent = `Final dealer total: ${state.dealer_total}.`;
    ui.values.hidden = true;
    if (scoredTable !== state.id) {
      for (const seat of state.seats) score[seat.result] += 1;
      scoredTable = state.id;
      renderScore();
    }
  }
}

function signed(value) { return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`; }

async function request(path, body = {}) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The table could not complete that move.");
  return data;
}

async function deal() {
  setBusy(true);
  try {
    selectedSeat = 0;
    render(await request("/api/multi/new", { seats: Number(ui.seatCount.value) }));
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function act(kind) {
  if (!table || table.stage !== "playing") return;
  // Route button actions to the current seat: turn order is enforced server-side.
  const seat = table.current_seat;
  setBusy(true);
  try {
    render(await request(`/api/multi/${kind}`, { id: table.id, seat }));
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

function setBusy(busy) {
  for (const button of [ui.deal, ui.hit, ui.stand, ui.ai]) button.disabled = busy;
  if (!busy && table) render(table);
}

function showError(error) {
  ui.toast.textContent = error.message;
  ui.toast.classList.add("visible");
  setTimeout(() => ui.toast.classList.remove("visible"), 3500);
}

ui.deal.addEventListener("click", deal);
ui.hit.addEventListener("click", () => act("hit"));
ui.stand.addEventListener("click", () => act("stand"));
ui.ai.addEventListener("click", () => act("ai"));
$("reset-score").addEventListener("click", () => { score = { win: 0, draw: 0, loss: 0 }; renderScore(); });
document.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (key === "h") act("hit");
  else if (key === "s") act("stand");
  else if (key === "a") act("ai");
  else if (key >= "1" && key <= "5" && table) {
    const seat = Number(key) - 1;
    if (seat < table.seats.length) { selectedSeat = seat; render(table); }
  }
});

renderScore();
