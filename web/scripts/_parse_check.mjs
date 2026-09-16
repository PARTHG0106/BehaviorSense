// Parse every page module as a real ES module. `node --check` treats a file as CommonJS and
// accepts things the browser rejects - it passed a stages.js whose template literal contained
// two double-quoted strings spanning line breaks, which would have been a blank page.
import { pathToFileURL } from "node:url";
import { readdirSync } from "node:fs";
import { join } from "node:path";

const dir = join(process.cwd(), "web", "scripts");
const files = readdirSync(dir).filter((f) => f.endsWith(".js"));
let bad = 0;
for (const f of files) {
  try {
    await import(pathToFileURL(join(dir, f)).href);
    console.log(`  OK   ${f}`);
  } catch (e) {
    // A module that parses but fails at import time (no `document`) is FINE here; only a
    // SyntaxError means the browser could not load it.
    if (e instanceof SyntaxError) { bad++; console.log(`  FAIL ${f}: ${e.message}`); }
    else console.log(`  OK   ${f}  (parsed; runtime: ${e.constructor.name})`);
  }
}
console.log(bad ? `${bad} module(s) FAILED to parse` : `all ${files.length} modules parse`);
process.exit(bad ? 1 : 0);
