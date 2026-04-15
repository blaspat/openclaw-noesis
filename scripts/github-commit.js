// github-commit.js — Create a verified commit via GitHub REST API
// No GPG/SSH signing needed. Commits show as "Verified" because they're
// authenticated via the GitHub Actions GITHUB_TOKEN.
//
// Usage: node scripts/github-commit.js <owner> <repo> <branch> <commit-message>
//
// The script bumps package.json + openclaw.plugin.json, then creates
// the commit via GitHub REST API. The bumped files are read directly
// from disk after bump-version.js has run.

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

function ghApi(method, path, body) {
  const cmd = body
    ? `gh api -X ${method} ${path} -f input='${JSON.stringify(body).replace(/'/g, "'\\''")}' --jq .sha`
    : `gh api -X ${method} ${path} --jq .sha`;
  return execSync(cmd, { encoding: "utf8" }).trim();
}

function ghApiRaw(method, path, body) {
  const cmd = body
    ? `gh api -X ${method} ${path} -f input='${JSON.stringify(body).replace(/'/g, "'\\''")}'`
    : `gh api -X ${method} ${path}`;
  execSync(cmd, { encoding: "utf8" });
}

try {
  // 1. Get current branch SHA
  const refData = JSON.parse(execSync(`gh api repos/${owner}/${repo}/git/refs/heads/${branch}`, { encoding: "utf8" }));
  const currentSha = refData.object.sha;

  // 2. Create blobs for the two bumped files
  const files = [
    { path: "package.json", content: fs.readFileSync(join(root, "package.json"), "utf8") },
    { path: "openclaw.plugin.json", content: fs.readFileSync(join(root, "openclaw.plugin.json"), "utf8") },
  ];

  const blobs = files.map((f) => {
    const sha = ghApi("POST", `repos/${owner}/${repo}/git/blobs`, {
      content: fs.readFileSync(join(root, f.path), "utf8"),
      encoding: "text",
    });
    return { path: f.path, sha };
  });

  // 3. Create tree
  const treeSha = ghApi("POST", `repos/${owner}/${repo}/git/trees`, {
    tree: blobs.map((b) => ({ path: b.path, mode: "100644", type: "blob", sha: b.sha })),
    base_tree: currentSha,
  });

  // 4. Create commit
  const commitSha = ghApi("POST", `repos/${owner}/${repo}/git/commits`, {
    message,
    tree: treeSha,
    parents: [currentSha],
    author: { name: "github-actions[bot]", email: "github-actions[bot]@users.noreply.github.com" },
  });

  // 5. Update branch ref
  ghApiRaw("PATCH", `repos/${owner}/${repo}/git/refs/heads/${branch}`, {
    sha: commitSha,
    force: false,
  });

  console.log(`GitHub API commit created: ${commitSha}`);
  process.exit(0);
} catch (err) {
  console.error("GitHub API commit failed:", err.message);
  process.exit(1);
}