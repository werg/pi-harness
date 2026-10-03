import { expect, test } from "vitest";
import { type Context, defineService, type JsonValue, type ReplicatedState } from "../src/index.ts";

type ReadonlyJson =
	| null
	| boolean
	| number
	| string
	| readonly ReadonlyJson[]
	| { readonly [key: string]: ReadonlyJson };

interface ReadonlyJsonService {
	readonly state: ReplicatedState<{ readonly entries: readonly ReadonlyJson[] }>;
	call(value: ReadonlyJson, context: Context): Promise<ReadonlyJson>;
}

interface MixedJsonService {
	call(value: JsonValue | ReadonlyJson, context: Context): Promise<{ readonly value?: ReadonlyJson }>;
}

interface InvalidReadonlyService {
	call(value: { readonly entries: readonly Date[] }, context: Context): Promise<void>;
}

test("accepts recursive readonly JSON contracts without unbounded type expansion", () => {
	expect(defineService<ReadonlyJsonService>("test.readonly-json").local).toBe(false);
	expect(defineService<MixedJsonService>("test.mixed-json").local).toBe(false);
	// @ts-expect-error A readonly container does not make Date a JSON value.
	expect(defineService<InvalidReadonlyService>("test.readonly-non-json").local).toBe(false);
});
