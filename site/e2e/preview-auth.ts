import type { BrowserContext } from "@playwright/test";

function previewCredentials() {
  const target = process.env.BASE_URL;
  const token = process.env.VERCEL_OIDC_TOKEN;
  return target && token ? { origin: new URL(target).origin, token } : null;
}

export async function authenticatePreview(context: BrowserContext) {
  const credentials = previewCredentials();
  if (!credentials) return;
  await context.route("**/*", async (route) => {
    if (new URL(route.request().url()).origin !== credentials.origin) {
      await route.continue();
      return;
    }
    await route.continue({
      headers: {
        ...route.request().headers(),
        "x-vercel-trusted-oidc-idp-token": credentials.token,
      },
    });
  });
}

export function previewRequestOptions(path: string) {
  const credentials = previewCredentials();
  if (
    !credentials ||
    new URL(path, credentials.origin).origin !== credentials.origin
  )
    return {};
  return {
    headers: { "x-vercel-trusted-oidc-idp-token": credentials.token },
    maxRedirects: 0,
  };
}
