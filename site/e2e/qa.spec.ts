import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mkdir } from "node:fs/promises";
const pages = [
  ["home", "/"],
  ["explorer", "/explorer/"],
  ["ablations", "/ablations/"],
  ["about", "/about/"],
];
for (const [name, route] of pages)
  for (const width of [1440, 390])
    for (const theme of ["light", "dark"])
      test(`QA ${name} ${width} ${theme}`, async ({ page }) => {
        await page.setViewportSize({ width, height: 1000 });
        await page.addInitScript(
          (t) => localStorage.setItem("counterprior-theme", t),
          theme,
        );
        await page.goto(route);
        await page.evaluate(() => document.fonts.ready);
        await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth,
          ),
        ).toBe(true);
        await mkdir("qa/screenshots", { recursive: true });
        await page.screenshot({
          path: `qa/screenshots/${name}-${width}-${theme}.png`,
          fullPage: true,
        });
        const a = await new AxeBuilder({ page }).analyze();
        expect(a.violations).toEqual([]);
      });
test("OG image is an actual page screenshot", async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 630 });
  await page.addInitScript(() =>
    localStorage.setItem("counterprior-theme", "light"),
  );
  await page.goto("/");
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: "public/opengraph.png" });
});
