import { test, expect } from "@playwright/test";
const routes = [
  ["/", "CounterPrior: Detecting"],
  ["/explorer/", "An answer is not the whole story."],
  ["/ablations/", "A reduction is not enough."],
  ["/about/", "About CounterPrior"],
];
for (const [route, heading] of routes)
  test(`${route} loads without console errors`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });
    const response = await page.goto(route);
    expect(response?.status()).toBe(200);
    await expect(page.getByRole("heading", { level: 1 })).toContainText(
      heading,
    );
    await expect(
      page.getByRole("navigation", { name: "Main navigation" }),
    ).toBeVisible();
    await page.waitForLoadState("networkidle");
    expect(errors).toEqual([]);
  });
test("explorer filters and resets without fabricating examples", async ({
  page,
}) => {
  await page.goto("/explorer/");
  await page.getByLabel("POPE split").selectOption("adversarial");
  await expect(page.getByLabel("POPE split")).toHaveValue("adversarial");
  for (const label of [
    "Baseline hallucinated",
    "Fixed by steering",
    "Broken by steering",
  ]) {
    const b = page.getByRole("button", { name: label, exact: true });
    await b.click();
    await expect(b).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("status")).toContainText("match your filters");
  }
  await page.getByRole("button", { name: "All outcomes", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "All outcomes", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
});
test("theme toggle works with keyboard and persists locally", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("counterprior-theme", "light"),
  );
  await page.goto("/");
  const toggle = page.getByRole("button", { name: "Switch to dark theme" });
  await toggle.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(
    page.getByRole("button", { name: "Switch to light theme" }),
  ).toBeVisible();
});
test("reproducibility assets are real files", async ({ request }) => {
  for (const path of [
    "/research.json",
    "/technical-report.pdf",
    "/citation.bib",
    "/sitemap.xml",
    "/robots.txt",
    "/favicon.svg",
  ]) {
    const response = await request.get(path);
    expect(response.status(), path).toBe(200);
    const bytes = await response.body();
    expect(bytes.length, path).toBeGreaterThan(0);
  }
  const response = await request.get("/research.json");
  const data = await response.json();
  if (data.status === "pending") {
    expect(data.examples).toEqual([]);
    expect(data.run).toBeNull();
  }
});
test("missing pages have a designed recovery", async ({ page }) => {
  const r = await page.goto("/a-path-that-does-not-exist/");
  expect(r?.status()).toBe(404);
  await expect(
    page.getByRole("heading", { name: "This path has no evidence." }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Return to the research" }),
  ).toHaveAttribute("href", "/");
});

test("internal navigation works on a plain static host", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Main navigation" });
  await nav.getByRole("link", { name: "Explorer", exact: true }).click();
  await expect(page).toHaveURL(/\/explorer\/$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "An answer is not the whole story.",
  );
  await nav.getByRole("link", { name: "Ablations", exact: true }).click();
  await expect(page).toHaveURL(/\/ablations\/$/);
  await nav.getByRole("link", { name: "About", exact: true }).click();
  await expect(page).toHaveURL(/\/about\/$/);
  await page.waitForLoadState("networkidle");
  expect(errors).toEqual([]);
});
