import fs from "fs";
import { execSync } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { createWriteStream } from "fs";
import { pipeline } from "stream/promises";

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

function apiRequest(method, url, body) {
  const tmpfile = `/tmp/gh-api-body-${Date.now()}.json`;
  if (body) fs.writeFileSync(tmpfile, JSON.stringify(body));

  const inputFlag = body ? `--data @${tmpfile}` : "";
  const cmd = `curl -s -X ${method} ${inputFlag} -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" -H "Accept: application/vnd.github+json" "${url}"`;

  const result = execSync(cmd, { encoding: "utf8" });
  try { fs.unlinkSync(tmpfile); } catch {}
  return JSON.parse(result);
}

try {
  // 1. Get current branch SHA
  const refData = apiRequest("GET", `https://api.github.com/repos/${owner}/${repo}/git/ref/heads/${branch}`);
  const currentSha = refData.object.sha;

  // 2. Read the bumped files
  const packageJson = fs.readFileSync(join(root, "package.json"), "utf8");
  const pluginJson = fs.readFileSync(join(root, "openclaw.plugin.json"), "utf8");

  // 3. Create blobs
  const blobPkg = apiRequest("POST", `https://api.github.com/repos/${owner}/${repo}/git/blobs`, {
    content: packageJson,
    encoding: "utf-8",
  });
  const blobPlugin = apiRequest("POST", `https://api.github.com/repos/${owner}/${repo}/git/blobs`, {
    content: pluginJson,
    encoding: "utf-8",
  });

  // 4. Create tree
  const treeSha = apiRequest("POST", `https://api.github.com/repos/${owner}/${repo}/git/trees`, {
    tree: [
      { path: "package.json", mode: "100644", type: "blob", sha: blobPkg.sha },
      { path: "openclaw.plugin.json", mode: "100644", type: "blob", sha: blobPlugin.sha },
    ],
    base_tree: currentSha,
  }).sha;

  // 5. Create commit
  const commitResponse = apiRequest("POST", `https://api.github.com/repos/${owner}/${repo}/git/commits`, {
    message,
    tree: treeSha,
    parents: [currentSha],
    author: {
      name: "github-actions[bot]",
      email: "github-actions[bot]@users.noreply.github.com",
    },
  });

  // 6. Update branch ref
  apiRequest("PATCH", `https://api.github.com/repos/${owner}/${repo}/git/refs/heads/${branch}`, {
    sha: commitResponse.sha,
    force: true,
  });

  console.log(`GitHub API commit created: ${commitResponse.sha}`);
  process.exit(0);
} catch (err) {
  console.error("GitHub API commit failed:", err.message);
  process.exit(1);
}