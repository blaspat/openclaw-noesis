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

const headers = {
  Authorization: `Bearer ${token}`,
  "Content-Type": "application/json",
  Accept: "application/vnd.github+json",
  "X-GitHub-Api-Version": "2022-11-28",
};

function curl(method, path, body) {
  const bodyStr = body ? JSON.stringify(body) : null;
  const cmd = bodyStr
    ? `curl -s -X ${method} -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" -H "Accept: application/vnd.github+json" -d @- https://api.github.com/${path}`
    : `curl -s -X ${method} -H "Authorization: Bearer ${token}" -H "Accept: application/vnd.github+json" https://api.github.com/${path}`;
  return JSON.parse(execSync(cmd, { input: bodyStr, encoding: "utf8" }));
}

try {
  // 1. Get current branch SHA
  const refData = curl("GET", `repos/${owner}/${repo}/git/ref/heads/${branch}`);
  const currentSha = refData.object.sha;

  // 2. Read the bumped files
  const packageJson = fs.readFileSync(join(root, "package.json"), "utf8");
  const pluginJson = fs.readFileSync(join(root, "openclaw.plugin.json"), "utf8");

  // 3. Create blobs
  const blobPkg = curl("POST", `repos/${owner}/${repo}/git/blobs`, {
    content: packageJson,
    encoding: "utf-8",
  });
  const blobPlugin = curl("POST", `repos/${owner}/${repo}/git/blobs`, {
    content: pluginJson,
    encoding: "utf-8",
  });

  // 4. Create tree
  const treeSha = curl("POST", `repos/${owner}/${repo}/git/trees`, {
    tree: [
      { path: "package.json", mode: "100644", type: "blob", sha: blobPkg.sha },
      { path: "openclaw.plugin.json", mode: "100644", type: "blob", sha: blobPlugin.sha },
    ],
    base_tree: currentSha,
  }).sha;

  // 5. Create commit
  const commitResponse = curl("POST", `repos/${owner}/${repo}/git/commits`, {
    message,
    tree: treeSha,
    parents: [currentSha],
    author: {
      name: "github-actions[bot]",
      email: "github-actions[bot]@users.noreply.github.com",
    },
  });

  // 6. Update branch ref
  curl("PATCH", `repos/${owner}/${repo}/git/refs/heads/${branch}`, {
    sha: commitResponse.sha,
    force: true,
  });

  console.log(`GitHub API commit created: ${commitResponse.sha}`);
  process.exit(0);
} catch (err) {
  console.error("GitHub API commit failed:", err.message || JSON.stringify(err));
  if (err.response) console.error("Response:", JSON.stringify(err.response, null, 2));
  process.exit(1);
}