/* Presentation of ACCEPTED claims only. Verification stays with the calling pipeline. */
import { getLanguage, onLanguageChange, t } from "./i18n.js";
import { SUMMARY, RECOMMENDATION } from "./fixtures.js";

const replayProse = Object.freeze({
  hi: {
    summary: "लगातार चौथे दिन, चलना-फिरना और सामाजिक संपर्क इस निवासी के अपने सामान्य स्तर से काफ़ी कम हैं। सामान्यतः तीन भोजन के बजाय एक भोजन दर्ज हुआ। यह पैटर्न केवल एक माप में नहीं, बल्कि कई स्वतंत्र मापों में दिखाई देता है।",
    recommendation: "आज फ़ोन करें। यदि कल तक चलना-फिरना सामान्य स्तर पर नहीं लौटता, तो जाँच की व्यवस्था करें।",
  },
  mr: {
    summary: "सलग चौथ्या दिवशी, हालचाल आणि सामाजिक संपर्क या रहिवाशाच्या स्वतःच्या नेहमीच्या पातळीपेक्षा खूप कमी आहेत. नेहमीच्या तीन जेवणांऐवजी एक जेवण नोंदवले गेले. हा कल फक्त एका मोजमापात नाही, तर अनेक स्वतंत्र मोजमापांत दिसतो.",
    recommendation: "आज फोन करा. उद्यापर्यंत चालणे नेहमीच्या पातळीवर परत आले नाही, तर तपासणीची व्यवस्था करा.",
  },
});

const countFeatures = new Set([
  "walking_bouts", "room_transitions", "sit_to_stand_count", "meal_events", "drinking_events",
  "medication_events", "phone_events", "fall_events",
]);
const secondsFeatures = new Set([
  "walking_duration_s", "mean_bout_duration_s", "mean_sit_to_stand_duration_s",
  "sitting_duration_s", "lying_duration_s", "standing_duration_s", "longest_inactive_block_s",
  "eating_duration_s", "cooking_duration_s", "tv_duration_s", "reading_duration_s",
  "social_interaction_duration_s", "housework_duration_s", "max_post_fall_immobility_s",
]);
const knownFeatures = new Set([...countFeatures, ...secondsFeatures, "visitor_count", "observed_hours",
  "tracking_coverage", "mobility_index", "sedentary_ratio"]);
const translations = new Map();
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const textValue = (value) => typeof value === "string" ? value : "";
const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));

function featureFor(claim, evidence) {
  const referenced = /^feat:([a-z0-9_]+):\d{4}-\d{2}-\d{2}$/.exec(textValue(claim.evidence_ref))?.[1];
  // The citation determines the feature. A display label must never relabel a measured value.
  return referenced || textValue(evidence?.feature);
}

function withUnit(value, feature, lang) {
  let unit = "";
  if (secondsFeatures.has(feature)) unit = "seconds";
  else if (countFeatures.has(feature)) unit = "count";
  else if (feature === "visitor_count") unit = "people";
  else if (feature === "observed_hours") unit = "hours";
  else if (feature === "mobility_index") unit = "score";
  else if (["tracking_coverage", "sedentary_ratio"].includes(feature)) unit = "ratio";
  // Do not round, convert to percentages, or infer an absent number.
  return `${String(value)}${unit ? ` ${t(`unit_${unit}`, {}, lang)}` : ""}`;
}

/** A factual localized reading of fields already accepted by the verifier. */
export function findingText(claim, evidence = {}, lang = getLanguage(), reliable = true) {
  if (lang === "en" && textValue(claim.text).trim()) return claim.text;
  const feature = featureFor(claim, evidence);
  const label = t(knownFeatures.has(feature) ? `feature_${feature}` : "unknownFeature", {}, lang);
  const parts = [finite(claim.claimed_value)
    ? `${label}: ${withUnit(claim.claimed_value, feature, lang)}.`
    : `${label}: ${t("missingValue", {}, lang)}`];
  if (reliable !== false) {
    const hasDirection = ["increase", "decrease", "unchanged"].includes(claim.direction);
    const hasChange = finite(claim.claimed_pct_change);
    if ((hasDirection || hasChange) && finite(evidence.baseline_median)) {
      parts.push(t("baseline", { value: withUnit(evidence.baseline_median, feature, lang) }, lang));
    }
    if (hasChange) parts.push(t("change", { value: String(claim.claimed_pct_change) }, lang));
    if (hasDirection) parts.push(t(`direction_${claim.direction}`, {}, lang));
  }
  return parts.join(" ");
}

function normalize(input) {
  const source = input && typeof input === "object" ? input : {};
  const claims = Array.isArray(source.claims)
    ? source.claims.filter((claim) => claim && typeof claim === "object").map((claim) => ({ ...claim })) : [];
  const evidence = Object.fromEntries(Object.entries(source.evidence || {})
    .filter(([, row]) => row && typeof row === "object").map(([ref, row]) => [ref, { ...row }]));
  const count = (value) => finite(value) && value >= 0 ? Math.floor(value) : null;
  const suppliedTotal = count(source.totalClaims);
  const suppliedWithheld = count(source.withheldClaims);
  const total = suppliedTotal ?? (claims.length + (suppliedWithheld ?? 0));
  const withheld = suppliedWithheld ?? Math.max(0, total - claims.length);
  return { ...source, summary: textValue(source.summary), recommendation: textValue(source.recommendation),
    day: textValue(source.day), claims, evidence, totalClaims: total, withheldClaims: withheld };
}

function originalMarkup(report, lang) {
  const claims = report.claims.map((claim) => `<li>${escape(findingText(claim,
    report.evidence[claim.evidence_ref], "en", report.isReliable))}</li>`).join("");
  return `<details class="caregiver__source" lang="en">
    <summary lang="${lang}">${escape(t("originalEnglish", {}, lang))}</summary>
    <div lang="en">
      <h4>Original summary</h4><p>${escape(report.summary || "No summary was provided.")}</p>
      <h4>Original recommendation</h4><p>${escape(report.recommendation || "No recommendation was provided.")}</p>
      <h4>Accepted findings</h4>${claims ? `<ul>${claims}</ul>` : "<p>No accepted findings.</p>"}
    </div>
  </details>`;
}

function markup(report, lang, prose, state) {
  const tr = (key, params) => t(key, params, lang);
  const notices = [];
  if (report.provenance === "demo") notices.push(tr("demoCaution"));
  else if (report.provenance === "simulated") notices.push(tr("simulatedCaution"));
  if (report.isReliable === false) notices.push(tr("clipCaution"));
  const findings = report.claims.map((claim) => {
    const row = report.evidence[claim.evidence_ref] || {};
    const date = /:(\d{4}-\d{2}-\d{2})$/.exec(textValue(claim.evidence_ref))?.[1];
    return `<li><p>${escape(findingText(claim, row, lang, report.isReliable))}</p>
      ${date ? `<small>${escape(tr("recordedOn", { day: date }))}</small>` : ""}</li>`;
  }).join("");
  const content = state === "ready"
    ? `<section class="caregiver__section"><h4>${escape(tr("summary"))}</h4><p>${escape(prose.summary || tr("noSummary"))}</p></section>
       <section class="caregiver__section"><h4>${escape(tr("recommendation"))}</h4><p>${escape(prose.recommendation || tr("noRecommendation"))}</p></section>`
    : `<p class="caregiver__status" role="status">${escape(tr(state === "pending" ? "translating" : "translationUnavailable"))}</p>
       ${state === "failed" ? `<button type="button" class="btn caregiver__retry" data-caregiver-retry>${escape(tr("retryTranslation"))}</button>` : ""}`;
  const escalation = report.escalate === true ? "escalateYes" : report.escalate === false ? "escalateNo" : "escalateUnknown";
  return `<article class="caregiver" lang="${lang}">
    <header class="caregiver__head"><h3>${escape(tr("caregiverTitle"))}</h3>
      ${report.day ? `<p class="caregiver__meta">${escape(tr("reportDay", { day: report.day }))}</p>` : ""}
      ${finite(report.observedHours) ? `<p class="caregiver__meta">${escape(tr("observedHours", { hours: String(report.observedHours) }))}</p>` : ""}
    </header>
    ${notices.map((notice) => `<p class="caregiver__notice caregiver__notice--caution">${escape(notice)}</p>`).join("")}
    ${content}
    <section class="caregiver__section"><h4>${escape(tr("findings"))}</h4>
      ${findings ? `<ul class="caregiver__findings">${findings}</ul>`
        : `<p class="caregiver__empty">${escape(tr(report.totalClaims === 0 ? "noScorable" : "noFindings"))}</p>`}
    </section>
    <p class="caregiver__counts">${escape(tr("counts", { accepted: report.claims.length,
      withheld: report.withheldClaims, total: report.totalClaims }))}</p>
    <p class="caregiver__escalation${report.escalate === true ? " caregiver__escalation--urgent" : ""}">${escape(tr(escalation))}</p>
    <p class="caregiver__scope">${escape(tr("verificationScope"))}</p>
    ${originalMarkup(report, lang)}
  </article>`;
}

async function translatedProse(report, lang, translate) {
  if (lang === "en" || (!report.summary && !report.recommendation)) return report;
  if (report.summary === SUMMARY && report.recommendation === RECOMMENDATION) return replayProse[lang];
  if (typeof translate !== "function") throw new Error("No translation service");
  const key = JSON.stringify([lang, report.summary, report.recommendation]);
  if (translations.has(key)) return translations.get(key);
  const pending = Promise.resolve().then(() => translate({ language: lang,
    summary: report.summary, recommendation: report.recommendation })).then((result) => {
    if (!result || typeof result.summary !== "string" || typeof result.recommendation !== "string"
      || (report.summary && !/[\u0900-\u097f]/u.test(result.summary))
      || (report.recommendation && !/[\u0900-\u097f]/u.test(result.recommendation))) {
      // An unchanged English response is not a translated fallback. Both supported target
      // languages use Devanagari; this catches absence of translation, not its correctness.
      throw new Error("Incomplete translation");
    }
    return { summary: result.summary, recommendation: result.recommendation };
  }).catch((error) => {
    if (translations.get(key) === pending) translations.delete(key);
    throw error;
  });
  if (translations.size >= 40) translations.delete(translations.keys().next().value);
  translations.set(key, pending);
  return pending;
}

/** translate({language, summary, recommendation}) resolves to translated prose only. */
export function mountCaregiverReport(element, { translate } = {}) {
  let report = null;
  let generation = 0;
  async function render() {
    if (!element || !report) return;
    const token = ++generation, snapshot = report, lang = getLanguage();
    const immediate = lang === "en" || (!snapshot.summary && !snapshot.recommendation)
      ? snapshot : snapshot.summary === SUMMARY && snapshot.recommendation === RECOMMENDATION ? replayProse[lang] : null;
    element.innerHTML = markup(snapshot, lang, immediate, immediate ? "ready" : "pending");
    if (immediate) return;
    try {
      const prose = await translatedProse(snapshot, lang, translate);
      if (token === generation && lang === getLanguage()) element.innerHTML = markup(snapshot, lang, prose, "ready");
    } catch {
      if (token === generation && lang === getLanguage()) element.innerHTML = markup(snapshot, lang, null, "failed");
    }
  }
  const unsubscribe = onLanguageChange(() => { void render(); });
  const retry = (event) => {
    if (event.target?.closest?.("[data-caregiver-retry]")) void render();
  };
  element?.addEventListener?.("click", retry);
  return {
    update(value) { report = normalize(value); return render(); },
    destroy() {
      generation++; unsubscribe(); report = null;
      element?.removeEventListener?.("click", retry);
    },
  };
}
