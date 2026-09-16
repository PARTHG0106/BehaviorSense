/* One analysed day, in the shape Agent 3 hands to Agent 4.
 *
 * These rows are the same fields `state_to_payload` puts in the prompt: the observed
 * value, the baseline median, the delta, the percentage change, the robust
 * z-score, and the direction. The model is given resolvable references rather than
 * asked to compose them, which is why C1 has never failed on a real run.
 *
 * The scenario is a resident with a genuine mobility decline — the case the system is
 * built to catch — so the claims below are about a real drop, not a synthetic one.
 */

export const DAY = "2026-02-22";

const rows = [
  ["walking_duration_s",             700,   1800,  -61.1, -4.82, "Walking duration"],
  ["sit_to_stand_count",               6,     14,  -57.1, -3.91, "Sit-to-stand transitions"],
  ["social_interaction_duration_s",  200,   1200,  -83.3, -3.44, "Time in conversation"],
  ["longest_inactive_block_s",     11400,   7200,   58.3,  3.12, "Longest inactive period"],
  ["room_transitions",                 9,     24,  -62.5, -3.05, "Room transitions"],
  ["meal_events",                      1,      3,  -66.7, -2.90, "Meals recorded"],
  ["medication_events",                0,      1, -100.0, -2.24, "Medication events"],
  ["lying_duration_s",             29400,  28000,    5.0,  0.41, "Time lying down"],
];

/** Evidence keyed by reference, exactly as the verifier resolves it. */
export const EVIDENCE = Object.fromEntries(rows.map(
  ([name, observed, baseline, pct, z, label]) => [
    `feat:${name}:${DAY}`,
    {
      ref: `feat:${name}:${DAY}`,
      feature: name,
      label,
      observed_value: observed,
      baseline_median: baseline,
      delta: observed - baseline,
      pct_change: pct,
      robust_z: z,
    },
  ],
));

export const SUMMARY =
  "Mobility and social contact are both well below this resident’s own baseline for a "
  + "fourth consecutive day, and one meal was recorded against a usual three. The pattern "
  + "is consistent across independent features rather than isolated to one.";

export const RECOMMENDATION =
  "Call today. If walking has not recovered by tomorrow, arrange a review.";

const ref = (name) => `feat:${name}:${DAY}`;

const unit = (name) => (name.endsWith("_s") ? " s" : "");
const val = (e) => `${e.observed_value.toLocaleString("en-GB")}${unit(e.feature)}`;
const base = (e) => `${e.baseline_median.toLocaleString("en-GB")}${unit(e.feature)}`;

/** The report as generated, before anything is injected. Six claims is the cap. */
export function baseClaims() {
  const at = (name) => EVIDENCE[ref(name)];
  const claim = (id, name, text) => {
    const e = at(name);
    return {
      claim_id: id,
      text,
      evidence_ref: e.ref,
      claimed_value: e.observed_value,
      claimed_pct_change: e.pct_change,
      direction: e.delta > 0 ? "increase" : e.delta < 0 ? "decrease" : "unchanged",
      fault: null,
    };
  };
  return [
    claim("c1", "walking_duration_s",
      "Walking duration fell to 700 s, against a rolling baseline of 1,800 s."),
    claim("c2", "sit_to_stand_count",
      "Sit-to-stand transitions dropped to 6, from a baseline of 14."),
    claim("c3", "social_interaction_duration_s",
      "Time in conversation fell to 200 s, well below the 1,200 s baseline."),
    claim("c4", "longest_inactive_block_s",
      "The longest unbroken inactive period rose to 11,400 s, up from 7,200 s."),
    claim("c5", "room_transitions",
      "Movement between rooms fell to 9, from a baseline of 24."),
    claim("c6", "meal_events",
      "1 meal was recorded, against a baseline of 3."),
  ];
}

/* The six faults the verifier claims to catch, applied as named transforms rather
 * than at random, so what you see is reproducible and each slip can say which fault
 * was injected into it. These mirror CORRUPTIONS in reporter.py.
 *
 * Each rewritten sentence is BUILT from the evidence row rather than patched with a
 * regex. A patched sentence kept its original direction words alongside the new ones
 * ("fell ... up from"), which the verifier correctly refuses to judge as either
 * direction — so the injected fault quietly failed to fire and the demo proved
 * nothing. An unfired fault is worse than a visible one.
 */
const FAULTS = {
  fabricated_ref: (c) => ({ ...c, evidence_ref: ref("stair_climbing_duration_s") }),

  invented_value: (c) => ({ ...c, claimed_value: c.claimed_value * 1.9 + 13 }),

  wrong_pct: (c) => ({ ...c, claimed_pct_change: c.claimed_pct_change + 47 }),

  // Right number, backwards story. C1-C3 all pass it; only C4 catches it.
  inverted_direction: (c) => {
    const e = EVIDENCE[c.evidence_ref];
    const flipped = c.direction === "decrease" ? "increase" : "decrease";
    const text = flipped === "increase"
      ? `${e.label} improved to ${val(e)}, up from a baseline of ${base(e)}.`
      : `${e.label} fell to ${val(e)}, down from a baseline of ${base(e)}.`;
    return { ...c, direction: flipped, text };
  },

  // The structured field stays correct and the prose contradicts it, so the claim is
  // internally inconsistent even before it is compared with the data.
  prose_contradiction: (c) => {
    const e = EVIDENCE[c.evidence_ref];
    const opposite = e.delta > 0 ? "decreased" : "increased";
    return { ...c, text: `${e.label} ${opposite} markedly, to ${val(e)}.` };
  },

  sign_flipped_pct: (c) => {
    const e = EVIDENCE[c.evidence_ref];
    const magnitude = Math.abs(c.claimed_pct_change);
    return {
      ...c,
      claimed_pct_change: magnitude,
      text: `${e.label} rose ${magnitude.toFixed(1)}% to ${val(e)}.`,
    };
  },
};

/* Fixed scripts rather than a seeded shuffle. The default view has to contain the
 * inverted-direction case, because that single claim — right number, backwards story,
 * passing C1 to C3 — is the argument for the whole verifier. Leaving it to chance
 * would mean the page sometimes fails to make its own point. */
const SCRIPTS = {
  none: [],
  some: [
    { index: 0, fault: "inverted_direction" },
    { index: 4, fault: "invented_value" },
  ],
  all: [
    { index: 0, fault: "inverted_direction" },
    { index: 1, fault: "invented_value" },
    { index: 2, fault: "wrong_pct" },
    { index: 3, fault: "prose_contradiction" },
    { index: 4, fault: "fabricated_ref" },
    { index: 5, fault: "sign_flipped_pct" },
  ],
};

export const SCENARIOS = { 0: "none", 0.34: "some", 1: "all" };

/** Apply a named script to the generated report. */
export function claimsFor(scenario) {
  const claims = baseClaims();
  for (const { index, fault } of SCRIPTS[scenario] ?? []) {
    claims[index] = { ...FAULTS[fault](claims[index]), fault };
  }
  return claims;
}


