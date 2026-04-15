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

const token = process.env.GH_TOKEN;
if (!token) {
  console.error("GH_TOKEN environment variable is required");
  process.exit(1);
}

const repoPath = `repos/${owner}/${repo}`;

function ghJson(method, path, body) {
  const bodyStr = JSON.stringify(body);
  const inputFile = `/tmp/gh-input-${Date.now()}.json`;
  fs.writeFileSync(inputFile, bodyStr);
  try {
    const out = execSync(
      `gh api -X ${method} ${path} --input ${inputFile}`,
      {
        encoding: "utf8",
        env: { ...process.env, GH_TOKEN: token },
      }
    ).trim();
    return JSON.parse(out);
  } finally {
    try { fs.unlinkSync(inputFile); } catch {}
  }
}

try {
  // 1. Get current branch SHA
  const refData = ghJson("GET", `${repoPath}/git/ref/heads/${branch}`, null);
  const currentSha = refData.object.sha;

  // 2. Read the bumped files
  const packageJson = fs.readFileSync(join(root, "package.json"), "utf8");
  const pluginJson = fs.readFileSync(join(root, "openclaw.plugin.json"), "utf8");

  // 3. Create blobs (need separate calls, --input doesn't work for parallel)
  const blobPkg = ghJson("POST", `${repoPath}/git/blobs`, {
    content: packageJson,
    encoding: "utf-8",
  });
  const blobPlugin = ghJson("POST", `${repoPath}/git/blobs`, {
    content: pluginJson,
    encoding: "utf-8",
  });

  // 4. Create tree
  const treeSha = ghJson("POST", `${repoPath}/git/trees`, {
    tree: [
      { path: "package.json", mode: "100644", type: "blob", sha: blobPkg.sha },
      { path: "openclaw.plugin.json", mode: "100644", type: "blob", sha: blobPlugin.sha },
    ],
    base_tree: currentSha,
  }).sha;

  // 5. Create commit
  const commitResponse = ghJson("POST", `${repoPath}/git/commits`, {
    message,
    tree: treeSha,
    parents: [currentSha],
    author: {
      name: "github-actions[bot]",
      email: "github-actions[bot]@users.noreply.github.com",
    },
  });

  // 6. Update branch ref
  ghJson("PATCH", `${repoPath}/git/refs/heads/${branch}`, {
    sha: commitResponse.sha,
    force: true,
  });

  console.log(`GitHub API commit created: ${commitResponse.sha}`);
  process.exit(0);
} catch (err) {
  console.error("GitHub API commit failed:", err.message);
  process.exit(1);
}