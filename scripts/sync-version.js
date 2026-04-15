#!/usr/bin/env node
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json'));
const plugin = JSON.parse(fs.readFileSync('openclaw.plugin.json'));
plugin.version = pkg.version;
fs.writeFileSync('openclaw.plugin.json', JSON.stringify(plugin, null, 2) + '\n');
console.log(`Synced openclaw.plugin.json → ${pkg.version}`);