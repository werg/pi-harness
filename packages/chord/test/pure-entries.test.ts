import { isJsonValue as jsonCheck, copyJson as jsonCopy } from "@earendil-works/chord/json";
import { replicatedState as stateFactory } from "@earendil-works/chord/state";
import { expect, test } from "vitest";
import { copyJson, isJsonValue, replicatedState } from "../src/index.ts";

test("standalone entries retain the root implementations", () => {
	expect(jsonCopy).toBe(copyJson);
	expect(jsonCheck).toBe(isJsonValue);
	expect(stateFactory).toBe(replicatedState);
});
