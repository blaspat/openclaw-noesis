import fs from 'fs';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(fs.readFileSync(join(__dirname, '..', 'package.json'), 'utf8'));
const plugin = JSON.parse(fs.readFileSync(join(__dirname, '..', 'openclaw.plugin.json'), 'utf8'));
plugin.version = pkg.version;
fs.writeFileSync(join(__dirname, '..', 'openclaw.plugin.json'), JSON.stringify(plugin, null, 2) + '\n');
console.log(`Synced openclaw.plugin.json → ${pkg.version}`);