#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(process.argv[2] ?? join(root, ".artifacts/vibestudio"));
const version = "1.1.0-vibestudio.1";
const packages = [
	["telemetry", "@earendil-works/pi-telemetry", "@panticonic/pi-telemetry"],
	["chord", "@earendil-works/chord", "@panticonic/pi-chord"],
	["ai", "@earendil-works/pi-ai", "@panticonic/pi-ai"],
	["durable", "@earendil-works/pi-durable", "@panticonic/pi-durable"],
];

function run(command, args, cwd = root) {
	return execFileSync(command, args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] });
}
function sha256(bytes) {
	return createHash("sha256").update(bytes).digest("hex");
}
function relocate(text) {
	for (const [, upstream, fork] of packages) text = text.replaceAll(upstream, fork);
	return text;
}
async function relocateTree(directory) {
	for (const entry of await readdir(directory, { withFileTypes: true })) {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) await relocateTree(path);
		else if (entry.isFile() && /\.(?:js|ts|map|json)$/.test(entry.name)) {
			const text = await readFile(path, "utf8");
			await writeFile(path, relocate(text));
		}
	}
}
function builtExports(value) {
	if (typeof value === "string") return value;
	return Object.fromEntries(
		Object.entries(value).filter(([key]) => key !== "source").map(([key, item]) => [key, builtExports(item)]),
	);
}

await mkdir(output, { recursive: true });
const paths = run("git", ["ls-files", "--cached", "--others", "--exclude-standard", "-z"])
	.split("\0").filter(Boolean).sort();
const inputs = [];
for (const path of paths) inputs.push({ path, sha256: sha256(await readFile(join(root, path))) });
// The offline model catalog is hydrated data, intentionally ignored by Git.
// It is nevertheless an input to the distributed provider package.
async function recordData(directory) {
	for (const entry of await readdir(directory, { withFileTypes: true })) {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) await recordData(path);
		else if (entry.isFile()) inputs.push({ path: relative(root, path), sha256: sha256(await readFile(path)) });
	}
}
await recordData(join(root, "packages/ai/src/providers/data"));
inputs.sort((a, b) => a.path.localeCompare(b.path));
const source = {
	upstreamCommit: run("git", ["rev-parse", "HEAD"]).trim(),
	adoptedUpstream: JSON.parse(await readFile(join(root, "UPSTREAM-ADOPTIONS.json"), "utf8")),
	inputDigest: sha256(JSON.stringify(inputs)),
	inputs,
	node: process.version,
	npm: run("npm", ["--version"]).trim(),
	typescript: run(process.execPath, [join(root, "node_modules/typescript/bin/tsc"), "--version"]).trim(),
	version,
};

// Compile the coherent source closure before assigning distribution identities.
// Published imports and declarations use the same exact fork package identities;
// no source condition, Git dependency, or runtime alias is needed by consumers.
const artifacts = [];
for (const [directory, upstream, name] of packages) {
	const packageRoot = join(root, "packages", directory);
	await rm(join(packageRoot, "dist"), { recursive: true, force: true });
	if (directory === "ai") process.stdout.write(run(process.execPath, ["--experimental-strip-types", "scripts/check-model-data.ts"], packageRoot));
	process.stdout.write(run(process.execPath, [join(root, "node_modules/typescript/bin/tsc"), "-p", "tsconfig.build.json"], packageRoot));
	if (directory === "ai") await cp(join(packageRoot, "src/providers/data"), join(packageRoot, "dist/providers/data"), { recursive: true });
	const manifest = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8"));
	const stage = await mkdtemp(join(output, `${directory}-stage-`));
	try {
		const files = ["dist", "README.md", "LICENSE", "SOURCE.json"];
		await cp(join(packageRoot, "dist"), join(stage, "dist"), { recursive: true });
		await relocateTree(join(stage, "dist"));
		await writeFile(join(stage, "README.md"), relocate(await readFile(join(packageRoot, "README.md"), "utf8")));
		if ((await readdir(packageRoot)).includes("CHANGELOG.md")) {
			await writeFile(join(stage, "CHANGELOG.md"), relocate(await readFile(join(packageRoot, "CHANGELOG.md"), "utf8")));
			files.push("CHANGELOG.md");
		}
		await cp(join(root, "LICENSE"), join(stage, "LICENSE"));
		const dependencies = Object.fromEntries(Object.entries(manifest.dependencies ?? {}).map(([dependency, range]) => {
			const fork = packages.find(([, original]) => original === dependency);
			return fork ? [fork[2], version] : [dependency, range];
		}));
		const { scripts: _scripts, devDependencies: _development, ...published } = manifest;
		await writeFile(join(stage, "package.json"), `${JSON.stringify({
			...published, name, version, dependencies, exports: builtExports(manifest.exports),
			files,
			vibestudio: { upstreamPackage: upstream, sourceDigest: source.inputDigest },
		}, null, 2)}\n`);
		await writeFile(join(stage, "SOURCE.json"), `${JSON.stringify(source, null, 2)}\n`);
		const [packed] = JSON.parse(run("npm", ["pack", "--ignore-scripts", "--json", "--pack-destination", output], stage));
		artifacts.push({ name, version, filename: packed.filename, integrity: packed.integrity,
			sha256: sha256(await readFile(join(output, packed.filename))), dependencies,
			files: packed.files.map(({ path, size }) => ({ path, size })) });
	} finally {
		await rm(stage, { recursive: true, force: true });
	}
}
const release = { source, artifacts };
await writeFile(join(output, "release.json"), `${JSON.stringify(release, null, 2)}\n`);
console.log(`Staged ${artifacts.length} built packages in ${relative(root, output)}; source ${source.inputDigest}`);
