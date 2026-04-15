/**
 * bump-version.js — Auto-bump the patch version and sync all version files.
 *
 * Usage: node scripts/bump-version.js [--major|--minor|--patch]
 *
 * Defaults to --patch if no flag given.
 * Reads current version from package.json, bumps it, syncs openclaw.plugin.json,
 * commits with a version tag, and prints the new version.
 *
 * Does NOT push — the caller (CI workflow) handles the push.
 */

import fs from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

const pkgPath = join(root, "package.json");
const pluginPath = join(root, "openclaw.plugin.json");

const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
const current = pkg.version; // e.g. "1.1.5"

const bumpType = process.argv.includes("--major")
  ? "major"
  : process.argv.includes("--minor")
  ? "minor"
  : "patch";

const [major, minor, patch] = current.split(".").map(Number);
const newVersion =
  bumpType === "major"
    ? `${major + 1}.0.0`
    : bumpType === "minor"
    ? `${major}.${minor + 1}.0`
    : `${major}.${minor}.${patch + 1}`;

// Update package.json
pkg.version = newVersion;
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");

// Sync openclaw.plugin.json
const plugin = JSON.parse(fs.readFileSync(pluginPath, "utf8"));
plugin.version = newVersion;
fs.writeFileSync(pluginPath, JSON.stringify(plugin, null, 2) + "\n");

console.log(`version: ${current} → ${newVersion} (${bumpType})`);
console.log(`::set-output name=NEW_VERSION::${newVersion}`);

export { newVersion };