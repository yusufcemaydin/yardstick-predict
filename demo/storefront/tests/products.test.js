import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { start } from "./helpers.js";

let app;
before(async () => {
  app = await start();
});
after(() => app.close());

test("the home page lists every product", async () => {
  const html = await (await fetch(`${app.url}/`)).text();
  for (const name of ["Desk lamp", "Notebook", "Mug", "Backpack"]) assert.match(html, new RegExp(name));
});

test("a product page shows its price", async () => {
  const res = await fetch(`${app.url}/products/mug`);
  assert.equal(res.status, 200);
  assert.match(await res.text(), /\$18/);
});

test("an unknown product is not found", async () => {
  const res = await fetch(`${app.url}/products/teapot`);
  assert.equal(res.status, 404);
});
