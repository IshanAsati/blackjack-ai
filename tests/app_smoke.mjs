/* Dev-only: exercises the SPA's count-table loop (bet_value + count_q + Kelly +
   splits) for 2000 rounds. Run: node tests/app_smoke.mjs */
import {mulberry32, countTag, mlpForward, encodeCount, Shoe, playRound, kellyUnits} from "../webapp/engine.js";
import {readFileSync} from "fs";

const D = (p) => JSON.parse(readFileSync(p, "utf8"));
const countQ = D("webapp/data/count_q.json");
const betValue = D("webapp/data/bet_value.json");
const pairs = new Set(D("webapp/data/pairs.json").splits.map(([r, u]) => `${r},${u}`));
const RULES = {decks: 6, penetration: 0.75, blackjack_payout: 1.5, double_after_split: true, ...betValue.rules};

const decide = (total, up, ace, tc, frac, first) => {
  const q = mlpForward(countQ.network, encodeCount(total, up, ace, tc, frac, first));
  if (!first) q[2] = -Infinity;
  return q.indexOf(Math.max(...q));
};
const rng = mulberry32(20260704);
let shoe = new Shoe(RULES, rng);
let rounds = 0, bets = 0, sized = 0, staked = 0, splits = 0, roll = 400;
const t0 = Date.now();
while (rounds < 2000) {
  if (!shoe.canDeal) shoe = new Shoe(RULES, rng);
  const {trueCount: tc, fraction} = shoe.observation();
  const edge = mlpForward(betValue.network, [Math.max(-15, Math.min(15, tc)) / 10, fraction])[0];
  const stake = kellyUnits(edge, roll, {kelly: 0.5, cap: 4, minEdge: 0.005});
  const drawn = [];
  const draw = () => { const c = shoe.cards.pop(); drawn.push(c); return c; };
  const fixed = {get: (k) => { const [t, u, a, f] = k.split(",").map(Number); return decide(t, u, a, tc, fraction, f); }};
  const r = playRound(shoe, draw, fixed, pairs, false);
  shoe.exposed += drawn.reduce((a, c) => a + countTag(c), 0);
  if (stake > 0) { bets++; staked += stake; sized += stake * r.reward; roll += stake * r.reward; }
  if (r.split) splits++;
  rounds++;
}
console.log(`rounds=${rounds} bets=${bets} splits=${splits} sized=${sized.toFixed(1)}u per100=${(sized / rounds * 100).toFixed(2)}u roll=${roll.toFixed(0)}u took=${Date.now() - t0}ms`);
if (!(bets > 0 && splits > 0 && Number.isFinite(sized))) { console.error("FAIL: loop produced no bets/splits"); process.exit(1); }
console.log("app smoke passed");
