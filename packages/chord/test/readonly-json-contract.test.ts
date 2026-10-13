import { Type } from "typebox";
import { expect, test } from "vitest";
import { type BehaviorOf, cell, copyJson, createHost, type InputOf, method, remote } from "../src/index.ts";

type ReadonlyJson =
	| null
	| boolean
	| number
	| string
	| readonly ReadonlyJson[]
	| { readonly [key: string]: ReadonlyJson };

const Json = Type.Unsafe<ReadonlyJson>(
	Type.Cyclic(
		{
			Json: Type.Union([
				Type.Null(),
				Type.Boolean(),
				Type.Number(),
				Type.String(),
				Type.Array(Type.Ref("Json")),
				Type.Record(Type.String(), Type.Ref("Json")),
			]),
		},
		"Json",
	),
);

const ReadonlyContract = remote("test.readonly-json", {
	version: 1,
	input: Json,
	variants: { json: { state: { value: Json }, methods: { echo: method(Json, Json) } } },
});

test("accepts recursive readonly JSON through schema contracts without unbounded type expansion", async () => {
	const value: ReadonlyJson = Object.freeze({
		entries: Object.freeze([null, true, 1, "text", Object.freeze({ nested: Object.freeze([2]) })]),
	});
	const input: InputOf<typeof ReadonlyContract> = value;
	const host = createHost({
		onError: (error) => {
			throw error;
		},
	});
	try {
		host.serve(ReadonlyContract, (initial, _principal, self): BehaviorOf<typeof ReadonlyContract> => {
			const state = self.state({ value: cell(Json, initial) });
			return { kind: "json", value: state.value, echo: (args) => args };
		});
		const handle = host.get(ReadonlyContract, input);
		await handle.ready;
		expect(handle.value.value).toEqual(value);
		expect(await handle.echo(value)).toEqual(value);
		handle.release();
	} finally {
		await host.close();
	}
});

test("readonly containers do not make Date a JSON value", () => {
	const invalid = Object.freeze({ entries: Object.freeze([new Date()]) });
	// @ts-expect-error A readonly container does not make Date a JSON value.
	const input: InputOf<typeof ReadonlyContract> = invalid;
	expect(() => copyJson(input)).toThrow(/strict JSON/);
});
