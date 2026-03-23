import { existsSync, readFileSync } from "fs";
import { querySystem } from "./query.js";

const DEFAULT_CASES = [
  {
    id: "core-routing",
    question: "How do I assign a JustCall number to a team member?",
    expectedFacet: "core",
  },
  {
    id: "email-routing",
    question: "How do I set up JustCall Email inbox and automation?",
    expectedFacet: "email",
  },
  {
    id: "workflow-question",
    question: "What are the steps to set up call forwarding?",
    expectedFacet: "core",
  },
];

function parseArgs() {
  const args = new Set(process.argv.slice(2));
  return {
    json: args.has("--json"),
    strict: args.has("--strict"),
    configPath: process.env.EVALS_CONFIG_PATH || "evals.cases.json",
  };
}

function loadCases(configPath) {
  if (!existsSync(configPath)) return DEFAULT_CASES;
  const parsed = JSON.parse(readFileSync(configPath, "utf-8"));
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error(`Expected non-empty array in ${configPath}`);
  }
  return parsed;
}

function hasCitation(answer) {
  return /\[#\d+\]/.test(answer);
}

function coreShareAmongTop(reranked, topN = 3) {
  if (!Array.isArray(reranked) || reranked.length === 0) return 0;
  const slice = reranked.slice(0, topN);
  const coreCount = slice.filter((item) => item.docFacet === "core").length;
  return coreCount / slice.length;
}

function evaluateCase(testCase, result, strict) {
  const checks = [];
  checks.push({
    name: "retrieval returns context",
    pass: !result.noContext && result.docs.length > 0,
  });
  checks.push({
    name: "answer is non-empty",
    pass: typeof result.answer === "string" && result.answer.trim().length > 0,
  });
  checks.push({
    name: "answer cites retrieved chunks",
    pass: hasCitation(result.answer),
  });

  if (testCase.expectedFacet) {
    checks.push({
      name: `facet router targets ${testCase.expectedFacet}`,
      pass: result.requestedFacet === testCase.expectedFacet,
    });
  }

  if ((testCase.expectedFacet || "").toLowerCase() === "core") {
    checks.push({
      name: "top retrieval mostly core docs",
      pass: coreShareAmongTop(result.reranked, 3) >= 0.66,
    });
  }

  if (Array.isArray(testCase.answerMustIncludeAny) && testCase.answerMustIncludeAny.length > 0) {
    const normalized = result.answer.toLowerCase();
    checks.push({
      name: "answer includes expected hint(s)",
      pass: testCase.answerMustIncludeAny.some((token) => normalized.includes(String(token).toLowerCase())),
    });
  }

  const passed = checks.filter((check) => check.pass).length;
  const failed = checks.length - passed;
  const criticalFailed = strict && checks.some((check) => !check.pass);
  return {
    id: testCase.id || testCase.question,
    question: testCase.question,
    requestedFacet: result.requestedFacet,
    topSource: result.reranked[0]?.doc?.metadata?.source ?? null,
    checks,
    passed,
    failed,
    criticalFailed,
  };
}

function printHumanSummary(results) {
  const totalChecks = results.reduce((acc, r) => acc + r.checks.length, 0);
  const passedChecks = results.reduce((acc, r) => acc + r.passed, 0);
  const failedChecks = totalChecks - passedChecks;

  console.log("\nFriendly eval report");
  console.log("====================");
  console.log(`Cases: ${results.length}`);
  console.log(`Checks passed: ${passedChecks}/${totalChecks}`);
  console.log(`Checks failed: ${failedChecks}`);

  for (const result of results) {
    console.log(`\nCase: ${result.id}`);
    console.log(`Question: ${result.question}`);
    if (result.topSource) {
      console.log(`Top source: ${result.topSource}`);
    }
    for (const check of result.checks) {
      console.log(` - [${check.pass ? "PASS" : "FAIL"}] ${check.name}`);
    }
  }
}

async function main() {
  const { json, strict, configPath } = parseArgs();
  const cases = loadCases(configPath);
  const results = [];

  for (const testCase of cases) {
    if (!testCase || typeof testCase.question !== "string" || testCase.question.trim().length === 0) {
      throw new Error(`Invalid eval case: ${JSON.stringify(testCase)}`);
    }
    const queryResult = await querySystem(testCase.question.trim());
    const evaluation = evaluateCase(testCase, queryResult, strict);
    results.push(evaluation);
  }

  if (json) {
    console.log(JSON.stringify({ results }, null, 2));
  } else {
    printHumanSummary(results);
  }

  const shouldFail = results.some((result) => result.criticalFailed || result.failed > 0);
  if (strict && shouldFail) {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error("Evals failed:", err.message);
  process.exit(1);
});
