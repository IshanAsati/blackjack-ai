/* Pure blackjack engine: no DOM. Ports of shoe.py / game.py / table.py logic.
   Card streams use a seeded local PRNG, so deals differ from the Python
   server's streams - mechanics and payouts are identical. */

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function handValue(cards) {
  let total = 0, aces = 0;
  for (const c of cards) { total += c === 1 ? 1 : c; if (c === 1) aces++; }
  let soft = false;
  while (aces > 0 && total + 10 <= 21) { total += 10; aces--; soft = true; }
  return {total, soft};
}

export function countTag(card) {
  if (card >= 2 && card <= 6) return 1;
  if (card === 1 || card === 10) return -1;
  return 0;
}

/* ---- neural nets (exact same math as torch Sequential) ---- */
export function mlpForward(net, input) {
  let x = input.slice();
  net.linears.forEach((layer, i) => {
    const y = new Array(layer.bias.length);
    for (let o = 0; o < layer.bias.length; o++) {
      let s = layer.bias[o];
      const row = layer.weights[o];
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      y[o] = s;
    }
    if (i >= net.activations.length) { x = y; return; } // output layer: linear
    const act = net.activations[i];
    x = act === "relu" ? y.map((v) => (v > 0 ? v : 0)) : y.map((v) => Math.tanh(v));
  });
  return x;
}

export function encodeClassic(total, dealer, ace) {
  const v = new Array(43).fill(0);
  v[total] = 1;
  v[32 + dealer - 1] = 1;
  v[42] = ace ? 1 : 0;
  return v;
}

export function encodeCount(total, dealer, ace, trueCount, fraction, canDouble) {
  const tc = Math.max(-15, Math.min(15, trueCount)) / 10;
  return [...encodeClassic(total, dealer, ace), tc, fraction, canDouble ? 1 : 0];
}

/* ---- finite shoe ---- */
const PATTERN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

export class Shoe {
  constructor(rules, rng) {
    this.rules = rules;
    this.cards = [];
    for (let d = 0; d < rules.decks; d++)
      for (let r = 0; r < 4; r++) this.cards.push(...PATTERN);
    for (let i = this.cards.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [this.cards[i], this.cards[j]] = [this.cards[j], this.cards[i]];
    }
    this.initialSize = this.cards.length;
    this.exposed = 0;
  }
  get canDeal() {
    return this.cards.length > Math.max(24, this.initialSize * (1 - this.rules.penetration));
  }
  observation() {
    const decksLeft = this.cards.length / 52;
    return {trueCount: this.exposed / Math.max(decksLeft, 0.25), fraction: this.cards.length / this.initialSize};
  }
}

function addCard(total, soft, card) {
  if (card === 1 && !soft && total + 11 <= 21) return {total: total + 11, soft: true};
  total += card === 1 ? 1 : card;
  if (total > 21 && soft) return {total: total - 10, soft: false};
  return {total, soft};
}

export function playRound(shoe, draw, fixed, pairSet, capture = false) {
  const p1 = draw(), up = draw(), p2 = draw(), hole = draw();
  const playerCards = [p1, p2], dealerCards = [up, hole], moves = [];
  let pt = addCard(0, false, p1); pt = addCard(pt.total, pt.soft, p2);
  let dt = addCard(0, false, up); dt = addCard(dt.total, dt.soft, hole);
  const playerNatural = pt.total === 21, dealerNatural = dt.total === 21;
  let stake = 1, reward, splitHands = null;
  if (dealerNatural) {
    reward = playerNatural ? 0 : -1;
  } else if (playerNatural) {
    reward = shoe.rules.blackjack_payout;
  } else if (p1 === p2 && pairSet.has(`${p1},${up}`)) {
    ({reward, stake, splitHands} = playSplit(shoe, draw, p1, up, dt, fixed, dealerCards, capture));
    playerCards.length = 0; playerCards.push(...splitHands[0].cards);
    moves.push(...splitHands[0].moves);
    pt = {total: splitHands[0].total, soft: splitHands[0].soft};
    dt = {total: splitHands[0].dealerTotal, soft: false};
  } else {
    let first = true;
    while (pt.total <= 21) {
      const action = fixed.get(`${pt.total},${up},${pt.soft ? 1 : 0},${first ? 1 : 0}`);
      if (capture) moves.push(["stand", "hit", "double"][action]);
      if (action === 0) break;
      const card = draw();
      if (capture) playerCards.push(card);
      pt = addCard(pt.total, pt.soft, card);
      if (action === 2) { stake = 2; break; }
      first = false;
    }
    if (pt.total > 21) {
      reward = -stake;
    } else {
      while (dt.total < 17) {
        const card = draw();
        if (capture) dealerCards.push(card);
        dt = addCard(dt.total, dt.soft, card);
      }
      reward = stake * (dt.total > 21 || pt.total > dt.total ? 1 : pt.total < dt.total ? -1 : 0);
    }
  }
  const round = {player: playerCards, dealer: dealerCards, player_total: pt.total,
    dealer_total: dt.total, moves, stake, natural: playerNatural, reward};
  if (splitHands) { round.split = true; round.hands = splitHands; }
  return round;
}

function playSplit(shoe, draw, rank, up, dt, fixed, dealerCards, capture) {
  const hands = [];
  for (let h = 0; h < 2; h++) {
    const card = draw();
    let total, soft;
    if (rank === 1) { const a = addCard(0, false, 1); const b = addCard(a.total, a.soft, card); total = b.total; soft = b.soft; }
    else { const a = addCard(0, false, rank); const b = addCard(a.total, a.soft, card); total = b.total; soft = b.soft; }
    hands.push({cards: [rank, card], total, soft, moves: [], stake: 1});
  }
  if (rank === 1) {
    for (const h of hands) if (capture) h.moves.push("stand");
  } else {
    for (const h of hands) {
      let first = true;
      while (h.total <= 21) {
        let action = fixed.get(`${h.total},${up},${h.soft ? 1 : 0},${first ? 1 : 0}`);
        if (action === 2 && (!first || !shoe.rules.double_after_split)) action = h.total === 21 ? 0 : 1;
        if (capture) h.moves.push(["stand", "hit", "double"][action]);
        if (action === 0) break;
        const card = draw();
        if (capture) h.cards.push(card);
        const t = addCard(h.total, h.soft, card);
        h.total = t.total; h.soft = t.soft;
        if (action === 2) { h.stake = 2; break; }
        first = false;
      }
    }
  }
  const live = hands.some((h) => h.total <= 21);
  if (live) while (dt.total < 17) {
    const card = draw();
    if (capture) dealerCards.push(card);
    dt = addCard(dt.total, dt.soft, card);
  }
  let reward = 0, stake = 0;
  for (const h of hands) {
    stake += h.stake;
    reward += h.total > 21 ? -h.stake : h.stake * (dt.total > 21 || h.total > dt.total ? 1 : h.total < dt.total ? -1 : 0);
    h.dealerTotal = dt.total;
  }
  return {reward, stake, splitHands: hands};
}

/* ---- unlimited-deck solo hand (classic) ---- */
const FACE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];
export function drawUnlimited(rng) {
  return FACE[Math.floor(rng() * FACE.length)];
}

export class ClassicGame {
  constructor(rng) {
    this.player = [drawUnlimited(rng), drawUnlimited(rng)];
    this.dealer = [drawUnlimited(rng), drawUnlimited(rng)];
    this.rng = rng;
    this.over = handValue(this.player).total > 21;
    this.result = this.over ? "loss" : null;
  }
  observation() {
    const {total, soft} = handValue(this.player);
    return {total, dealer: this.dealer[0], ace: soft ? 1 : 0};
  }
  hit() {
    this.player.push(drawUnlimited(this.rng));
    if (handValue(this.player).total > 21) { this.over = true; this.result = "loss"; }
  }
  stand() {
    while (handValue(this.dealer).total < 17) this.dealer.push(drawUnlimited(this.rng));
    const p = handValue(this.player).total, d = handValue(this.dealer).total;
    this.result = d > 21 || p > d ? "win" : p < d ? "loss" : "draw";
    this.over = true;
  }
}

/* ---- Kelly sizing (port of sizing.py) ---- */
export function kellyUnits(edge, roll, {kelly = 0.5, variance = 1.3, cap = 4, minEdge = 0.005} = {}) {
  if (edge < minEdge) return 0;
  return Math.max(1, Math.min(cap, Math.floor((kelly * edge * roll) / variance)));
}
