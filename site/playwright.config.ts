import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 1,
  reporter: [
    ["list"],
    [
      "json",
      {
        outputFile: process.env.QA_REPORT_PATH || "qa/playwright-results.json",
      },
    ],
    ["html", { open: "never" }],
  ],
  use: {
    baseURL: process.env.BASE_URL || "http://127.0.0.1:4173",
    // Authenticated traces can contain credential-bearing network headers.
    trace:
      process.env.BASE_URL && process.env.VERCEL_OIDC_TOKEN
        ? "off"
        : "retain-on-failure",
    reducedMotion: "reduce",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    {
      name: "firefox",
      testIgnore: "**/qa.spec.ts",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      testIgnore: "**/qa.spec.ts",
      use: { ...devices["Desktop Safari"] },
    },
  ],
  webServer: process.env.BASE_URL
    ? undefined
    : {
        command: "npm run start",
        url: "http://127.0.0.1:4173",
        reuseExistingServer: !process.env.CI,
        timeout: 120000,
      },
});
