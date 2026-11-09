import { test, expect } from "@playwright/test";
import { authenticatePreview } from "./preview-auth";
test.beforeEach(async ({ context }) => authenticatePreview(context));
import AxeBuilder from "@axe-core/playwright";
import { mkdir } from "node:fs/promises";
const pages = [
  ["home", "/"],
  ["record", "/record/"],
  ["explorer", "/explorer/"],
  ["ablations", "/ablations/"],
  ["about", "/about/"],
];
for (const [name, route] of pages)
  for (const width of [1440, 360])
      test(`QA ${name} ${width}`, async ({ page }) => {
        await page.setViewportSize({ width, height: 1000 });
        await page.goto(route);
        await page.evaluate(() => document.fonts.ready);
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth,
          ),
        ).toBe(true);
        await mkdir("qa/screenshots", { recursive: true });
        await page.screenshot({
          path: `qa/screenshots/${name}-${width}.png`,
          fullPage: true,
        });
        const a = await new AxeBuilder({ page }).analyze();
        expect(a.violations).toEqual([]);
      });
