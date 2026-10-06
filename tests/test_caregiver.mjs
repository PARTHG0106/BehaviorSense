import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { findingText, mountCaregiverReport } from "../web/scripts/caregiver.js";
import { getLanguage, initI18n, setLanguage, t } from "../web/scripts/i18n.js";
import { SUMMARY, RECOMMENDATION, baseClaims, EVIDENCE } from "../web/scripts/fixtures.js";

const tick = () => new Promise((resolve) => setImmediate(resolve));
const claim = { claim_id: "accepted", text: "Walking duration was 0.074 seconds.",
  evidence_ref: "feat:walking_duration_s:2026-09-13", claimed_value: 0.074,
  claimed_pct_change: null, direction: null };
const evidence = { [claim.evidence_ref]: { feature: "walking_duration_s", observed_value: 0.074,
  baseline_median: null, pct_change: null } };
const report = { summary: "Observed a short walk.", recommendation: "Review the recorded activities.",
  claims: [claim], evidence, totalClaims: 2, withheldClaims: 1, escalate: false,
  provenance: "simulated", observedHours: 0.166, isReliable: false };
const sourceStart = (html) => html.indexOf('<details class="caregiver__source"');
const primary = (html) => html.slice(0, sourceStart(html));

test("caregiver report language, evidence, and asynchronous presentation", async (suite) => {
  await suite.test("daily header and tallies have localized text nodes", () => {
    const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
    for (const key of ["dailyReport", "resident", "totalClaimsLabel", "shownClaimsLabel", "withheldClaimsLabel"]) {
      assert.ok(html.includes(`data-i18n="${key}"`));
      for (const lang of ["hi", "mr"]) assert.match(t(key, {}, lang), /[\u0900-\u097f]/u);
    }
    assert.match(html, /<span data-i18n="resident">Resident<\/span> &middot;/);
    assert.match(html, /<html[^>]*lang="en"/);
  });

  await suite.test("preferences persist best-effort and keep the source document English", () => {
    const stored = new Map([["behaviorsense.report-language", "mr"]]);
    globalThis.localStorage = { getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value) };
    const label = { dataset: { i18n: "languageLabel" }, textContent: "", lang: "" };
    const placeholder = { dataset: { i18nPlaceholder: "enrolPlaceholder" }, setAttribute(key, value) { this[key] = value; } };
    const select = { options: [], replaceChildren(...nodes) { this.options = nodes; }, addEventListener() {} };
    globalThis.document = { documentElement: { lang: "en" },
      querySelectorAll: (selector) => selector === "[data-i18n]" ? [label] : [placeholder],
      getElementById: () => select, createElement: () => ({}) };
    assert.equal(initI18n(), "mr");
    assert.equal(label.lang, "mr");
    assert.equal(placeholder.lang, "mr");
    assert.deepEqual(select.options.map((option) => option.textContent), ["English", "हिन्दी", "मराठी"]);
    assert.equal(document.documentElement.lang, "en");
    setLanguage("hi");
    assert.equal(stored.get("behaviorsense.report-language"), "hi");
    assert.equal(setLanguage("unrecognized"), false);
    assert.equal(getLanguage(), "hi");
    assert.equal(t("missingDictionaryKey"), "missingDictionaryKey");
    globalThis.localStorage = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
    assert.doesNotThrow(() => { initI18n(); setLanguage("en"); });
    delete globalThis.document;
    delete globalThis.localStorage;
  });

  await suite.test("translated findings preserve source numbers and never invent comparisons", () => {
    for (const lang of ["hi", "mr"]) {
      const text = findingText(claim, evidence[claim.evidence_ref], lang, false);
      assert.match(text, /0\.074/);
      assert.doesNotMatch(text, /0\.07[^4]/);
      assert.doesNotMatch(text, /%|null|undefined|walking_duration_s/);
      assert.ok(!text.includes(t("baseline", { value: "" }, lang)));
      const unknown = findingText({ evidence_ref: "feat:unknown_metric:2026-09-13", claimed_value: null }, {}, lang);
      assert.ok(unknown.includes(t("unknownFeature", {}, lang)));
      assert.ok(unknown.includes(t("missingValue", {}, lang)));
      assert.doesNotMatch(unknown, /NaN|undefined|0/);
      const counts = findingText({ evidence_ref: "feat:meal_events:2026-09-13", claimed_value: 0 }, {}, lang);
      assert.match(counts, /0/);
      const comparison = findingText({ ...claim, claimed_value: 700, claimed_pct_change: -61.1, direction: "decrease" },
        { baseline_median: 1800 }, lang, true);
      assert.match(comparison, /700/); assert.match(comparison, /1800/); assert.match(comparison, /-61\.1%/);
      const withheld = findingText({ ...claim, claimed_pct_change: -61.1, direction: "decrease" },
        { baseline_median: 1800 }, lang, false);
      assert.doesNotMatch(withheld, /1800|61\.1/);
    }
  });

  await suite.test("offline replay has translated prose and accepted-only original disclosure", async () => {
    setLanguage("hi");
    const element = { innerHTML: "" };
    const instance = mountCaregiverReport(element, { translate: () => { throw new Error("offline"); } });
    const input = { summary: SUMMARY, recommendation: RECOMMENDATION, claims: baseClaims().slice(0, 1),
      evidence: EVIDENCE, totalClaims: 6, withheldClaims: 5, provenance: "demo" };
    const before = JSON.stringify(input);
    await instance.update(input);
    assert.equal(JSON.stringify(input), before);
    assert.ok(primary(element.innerHTML).includes("लगातार चौथे दिन"));
    assert.ok(!primary(element.innerHTML).includes(SUMMARY));
    assert.ok(element.innerHTML.includes(SUMMARY));
    assert.match(element.innerHTML, /<details class="caregiver__source" lang="en">/);
    assert.ok(primary(element.innerHTML).includes("5 रोके गए"));
    assert.doesNotMatch(element.innerHTML, /Movement between rooms fell to 9/);
    setLanguage("mr");
    await tick();
    assert.ok(primary(element.innerHTML).includes("सलग चौथ्या दिवशी"));
    assert.ok(!primary(element.innerHTML).includes("लगातार चौथे दिन"));
    instance.destroy();
  });

  await suite.test("old language and old report responses cannot overwrite current report", async () => {
    setLanguage("hi");
    const pending = [];
    const element = { innerHTML: "" };
    const instance = mountCaregiverReport(element, { translate: (input) => new Promise((resolve) => pending.push({ input, resolve })) });
    const first = instance.update(report);
    await tick();
    setLanguage("mr");
    await tick();
    assert.equal(pending.length, 2);
    pending[1].resolve({ summary: "मराठी सारांश", recommendation: "मराठी सूचना" });
    await tick();
    pending[0].resolve({ summary: "हिन्दी पुराना सारांश", recommendation: "हिन्दी पुरानी सलाह" });
    await first;
    assert.ok(primary(element.innerHTML).includes("मराठी सारांश"));
    assert.ok(!primary(element.innerHTML).includes("हिन्दी पुराना सारांश"));
    const second = instance.update({ ...report, summary: "Another unique source." });
    await tick();
    const third = instance.update({ ...report, summary: "Newest unique source." });
    await tick();
    pending[3].resolve({ summary: "नवीन अहवाल", recommendation: "नवीन सूचना" });
    await third;
    pending[2].resolve({ summary: "जुना अहवाल", recommendation: "जुनी सूचना" });
    await second;
    assert.ok(primary(element.innerHTML).includes("नवीन अहवाल"));
    assert.ok(!primary(element.innerHTML).includes("जुना अहवाल"));
    instance.destroy();
  });

  await suite.test("failed translation keeps localized facts without leaking English prose", async () => {
    setLanguage("hi");
    const element = { innerHTML: "" };
    const instance = mountCaregiverReport(element, { translate: async () => { throw new Error("private upstream error"); } });
    await instance.update({ ...report, summary: "Unique failed English summary." });
    assert.ok(primary(element.innerHTML).includes(t("translationUnavailable")));
    assert.ok(primary(element.innerHTML).includes("0.074"));
    assert.ok(primary(element.innerHTML).includes(t("simulatedCaution")));
    assert.ok(primary(element.innerHTML).includes(t("clipCaution")));
    assert.ok(!primary(element.innerHTML).includes("Unique failed English summary."));
    assert.ok(!element.innerHTML.includes("private upstream error"));
    assert.ok(element.innerHTML.includes("Unique failed English summary."));
    await instance.update({ claims: [], totalClaims: 0 });
    assert.ok(primary(element.innerHTML).includes(t("noScorable")));
    await instance.update({ claims: [], totalClaims: 3, withheldClaims: 3 });
    assert.ok(primary(element.innerHTML).includes(t("noFindings")));
    instance.destroy();
  });

  await suite.test("source and translated content are escaped and source objects stay unchanged", async () => {
    setLanguage("en");
    const element = { innerHTML: "" };
    const instance = mountCaregiverReport(element, { translate: async () => ({ summary: 'सारांश <img src=x onerror="alert(1)">', recommendation: "सुझाव <script>bad()</script>" }) });
    const attackClaim = Object.freeze({ ...claim, text: "<img src=x onerror=attack()>" });
    const input = Object.freeze({ ...report, summary: "<script>attack()</script>", claims: Object.freeze([attackClaim]),
      evidence: Object.freeze({ [claim.evidence_ref]: Object.freeze({ ...evidence[claim.evidence_ref] }) }) });
    const before = JSON.stringify(input);
    await instance.update(input);
    assert.doesNotMatch(element.innerHTML, /<script>|<img/);
    assert.match(element.innerHTML, /&lt;script&gt;/);
    setLanguage("hi");
    await tick();
    await tick();
    assert.doesNotMatch(element.innerHTML, /<script>|<img/);
    assert.match(element.innerHTML, /&lt;img/);
    assert.equal(JSON.stringify(input), before);
    instance.destroy();
    setLanguage("en");
  });

  await suite.test("untranslated successful responses are an explicit unavailable state", async () => {
    setLanguage("mr");
    const element = { innerHTML: "" };
    const instance = mountCaregiverReport(element, { translate: async ({ summary, recommendation }) => ({ summary, recommendation }) });
    await instance.update({ ...report, summary: "This endpoint returned unchanged English." });
    assert.ok(primary(element.innerHTML).includes(t("translationUnavailable")));
    assert.ok(!primary(element.innerHTML).includes("This endpoint returned unchanged English."));
    assert.ok(primary(element.innerHTML).includes("0.074"));
    instance.destroy();
    setLanguage("en");
  });

  await suite.test("failed translation can be retried in the selected language and listeners are cleaned up", async () => {
    setLanguage("mr");
    const listeners = new Map();
    const element = { innerHTML: "", addEventListener(name, fn) { listeners.set(name, fn); },
      removeEventListener(name, fn) { if (listeners.get(name) === fn) listeners.delete(name); } };
    let calls = 0;
    const instance = mountCaregiverReport(element, { translate: async ({ language }) => {
      assert.equal(language, "mr");
      if (++calls === 1) throw new Error("temporary failure");
      return { summary: "पुन्हा तयार झालेला सारांश", recommendation: "मराठी सूचना" };
    } });
    await instance.update({ ...report, summary: "A unique source for retry." });
    assert.match(primary(element.innerHTML), /data-caregiver-retry/);
    assert.ok(primary(element.innerHTML).includes(t("retryTranslation")));
    listeners.get("click")({ target: { closest: () => ({}) } });
    await tick();
    assert.equal(calls, 2);
    assert.ok(primary(element.innerHTML).includes("पुन्हा तयार झालेला सारांश"));
    assert.doesNotMatch(primary(element.innerHTML), /data-caregiver-retry/);
    instance.destroy();
    assert.equal(listeners.size, 0);
    setLanguage("en");
  });
});
