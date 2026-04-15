import fs from "fs";
import { execSync } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

const [, , owner, repo, branch, ...msgParts] = process.argv;
const message = msgParts.join(" ");

if (!owner || !repo || !branch || !message) {
  console.error("Usage: node scripts/github-commit.js <owner> <repo> <branch> <message>");
  process.exit(1);
}

function ghApiJson(method, path, body) {
  const bodyStr = JSON.stringify(body);
  const encoded = Buffer.from(bodyStr).toString("base64");
  const cmd = `gh api -X ${method} ${path} -f input='${encoded}'`;
  return JSON.parse(execSync(cmd, { encoding: "utf8" }));
}

function ghApiSha(method, path, body) {
  return ghApiJson(method, path, body).sha;
}

try {
  const currentSha = ghApiJson("GET", `repos/${owner}/${repo}/git/refs/heads/${branch}`).object.sha;

  const files = [
    { path: "package.json", content: fs.readFileSync(join(root, "package.json"), "utf8") },
    { path: "openclaw.plugin.json", content: fs.readFileSync(join(root, "openclaw.plugin.json"), "utf8") },
  ];

  const blobs = files.map((f) => {
    const sha = ghApiSha("POST", `repos/${owner}/${repo}/git/blobs`, {
      content: fs.readFileSync(join(root, f.path), "utf8"),
      encoding: "text",
    });
    return { path: f.path, sha };
  });

  const treeSha = ghApiSha("POST", `repos/${owner}/${repo}/git/trees`, {
    tree: blobs.map((b) => ({ path: b.path, mode: "100644", type: "blob", sha: b.sha })),
    base_tree: currentSha,
  });

  const commitResponse = ghApiJson("POST", `repos/${owner}/${repo}/git/commits`, {
    message,
    tree: treeSha,
    parents: [currentSha],
    author: { name: "github-actions[bot]", email: "github-actions[bot]@users.noreply.github.com" },
  });

  ghApiJson("PATCH", `repos/${owner}/${repo}/git/refs/heads/${branch}`, {
    sha: commitResponse.sha,
    force: true,
  });

  console.log(`GitHub API commit created: ${commitResponse.sha}`);
  process.exit(0);
} catch (err) {
  console.error("GitHub API commit failed:", err.message);
  process.exit(1);
}