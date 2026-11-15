#!/usr/bin/env node
// A local npm registry for an unpublished Vibestudio Pi release: it serves the metadata and tarballs that
// `scripts/package-vibestudio.mjs` staged on top of the real registry (every other package passes through).
//
//   node scripts/serve-vibestudio-registry.mjs <staged dir> [port]
//   npm_config_registry=http://127.0.0.1:<port>/ pnpm type-check:userland ...
//
// It is a stand-in for publishing: publishing the staged tarballs makes it unnecessary.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const directory = resolve(process.argv[2] ?? ".artifacts/vibestudio");
const port = Number(process.argv[3] ?? 4873);
const upstream = process.env.VIBESTUDIO_UPSTREAM_REGISTRY ?? "https://registry.npmjs.org";
const release = JSON.parse(await readFile(join(directory, "release.json"), "utf8"));

const local = new Map();
for (const artifact of release.artifacts) {
	const bytes = await readFile(join(directory, artifact.filename));
	const manifest = JSON.parse(
		execFileSync("tar", ["xzOf", join(directory, artifact.filename), "package/package.json"], { encoding: "utf8" }),
	);
	local.set(artifact.name, {
		artifact,
		bytes,
		manifest,
		shasum: createHash("sha1").update(bytes).digest("hex"),
	});
}

function packument(entry, upstreamDocument) {
	const base = `http://127.0.0.1:${port}`;
	const version = {
		...entry.manifest,
		_id: `${entry.artifact.name}@${entry.artifact.version}`,
		dist: {
			tarball: `${base}/tarballs/${entry.artifact.filename}`,
			integrity: entry.artifact.integrity,
			shasum: entry.shasum,
		},
	};
	const document = upstreamDocument ?? { name: entry.artifact.name, versions: {}, "dist-tags": {} };
	return {
		...document,
		versions: { ...document.versions, [entry.artifact.version]: version },
		time: { ...document.time, [entry.artifact.version]: new Date().toISOString() },
	};
}

createServer(async (request, response) => {
	try {
		const url = new URL(request.url ?? "/", "http://local");
		if (url.pathname.startsWith("/tarballs/")) {
			const file = decodeURIComponent(url.pathname.slice("/tarballs/".length));
			const entry = [...local.values()].find((item) => item.artifact.filename === file);
			if (!entry) {
				response.writeHead(404).end();
				return;
			}
			response.writeHead(200, { "content-type": "application/octet-stream", "content-length": entry.bytes.length });
			response.end(entry.bytes);
			return;
		}
		const name = decodeURIComponent(url.pathname.slice(1));
		const entry = local.get(name);
		const headers = { accept: request.headers.accept ?? "application/json" };
		const remote = await fetch(`${upstream}${url.pathname}${url.search}`, { headers });
		if (!entry) {
			response.writeHead(remote.status, { "content-type": remote.headers.get("content-type") ?? "application/json" });
			response.end(Buffer.from(await remote.arrayBuffer()));
			return;
		}
		const document = remote.ok ? await remote.json() : undefined;
		response.writeHead(200, { "content-type": "application/json" });
		response.end(JSON.stringify(packument(entry, document)));
	} catch (error) {
		response.writeHead(502, { "content-type": "text/plain" });
		response.end(String(error));
	}
}).listen(port, "127.0.0.1", () => {
	console.log(`vibestudio pi registry on http://127.0.0.1:${port} for ${[...local.keys()].join(", ")}`);
});
