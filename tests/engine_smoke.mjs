/* Dev-only engine cross-check: node tests/engine_smoke.mjs (not part of pytest). */
import {mulberry32, handValue, countTag, mlpForward, encodeClassic, encodeCount,
  Shoe, playRound, ClassicGame, drawUnlimited, kellyUnits} from "../webapp/engine.js";
import {readFileSync} from "fs";

const assert = (cond, msg) => { if (!cond) { console.error("FAIL:", msg); process.exit(1); } };
// hand values incl. soft aces
assert(handValue([1, 7]).total === 18 && handValue([1, 7]).soft, "soft 18");
assert(handValue([1, 7, 8]).total === 16 && !handValue([1, 7, 8]).soft, "hard 16");
assert(handValue([10, 10, 5]).total === 25, "bust");
// Hi-Lo tags mirror shoe.count_tag
assert(countTag(5) === 1 && countTag(10) === -1 && countTag(1) === -1 && countTag(8) === 0, "tags");
// encodings: shapes + one-hot positions
const e = encodeClassic(16, 10, 0);
assert(e.length === 43 && e[16] === 1 && e[41] === 1 && e[42] === 0, "classic encoding");
const c = encodeCount(16, 10, 0, 20, 0.5, true);
assert(c.length === 46 && c[43] === 1.5 && c[44] === 0.5 && c[45] === 1, "count encoding + clipping");
// exact net math on shipped weights vs python-computed reference (see test_export.py)
const classic = JSON.parse(readFileSync("webapp/data/classic_q.json", "utf8"));
const q = mlpForward(classic.network, encodeClassic(16, 10, 0));
console.log("16v10 Q:", q.map((v) => v.toFixed(4)).join(" "), "->", q[1] > q[0] ? "hit" : "stand");
// Pinned cross-check: python torch gives [-0.6839, -0.6001] (catches activation bugs).
assert(Math.abs(q[0] + 0.6839) < 0.001 && Math.abs(q[1] + 0.6001) < 0.001, "Q values match torch");
// Kelly sizing mirrors sizing.kelly_units
assert(kellyUnits(0.001, 400, {cap: 4, minEdge: 0.005}) === 0, "kelly gate");
assert(kellyUnits(0.02, 400, {cap: 4, minEdge: 0.005}) === 3, "kelly 3u");
assert(kellyUnits(0.2, 400, {cap: 4, minEdge: 0.005}) === 4, "kelly cap");
// shoe mechanics: full deck, penetration cutoff, observation shape
const shoe = new Shoe({decks: 6, penetration: 0.75, blackjack_payout: 1.5, double_after_split: true}, mulberry32(11));
assert(shoe.cards.length === 312 && shoe.canDeal, "full shoe");
const o = shoe.observation();
assert(Math.abs(o.trueCount) < 1e-9 && o.fraction === 1, "fresh shoe count");
// deterministic round: same seed+rng state -> same reward path twice
const fixed = new Map([["16,10,0,1", 1], ["16,10,0,0", 1]]);
const runOnce = () => {
  const s = new Shoe({decks: 6, penetration: 0.75, blackjack_payout: 1.5, double_after_split: true}, mulberry32(5));
  const drawn = [];
  const draw = () => { const c = s.cards.pop(); drawn.push(c); return c; };
  const r = playRound(s, draw, fixed, new Set(), true);
  return JSON.stringify([r.reward, r.player, r.dealer]);
};
assert(runOnce() === runOnce(), "deterministic rounds");
// classic game resolves
const g = new ClassicGame(mulberry32(3));
while (!g.over) g.observation().total < 17 ? g.hit() : g.stand();
assert(["win", "loss", "draw"].includes(g.result), "classic resolves");
console.log("engine smoke passed");
