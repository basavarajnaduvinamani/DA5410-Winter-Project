import type { MetadataRoute } from "next";
export const dynamic = "force-static";
export default function sitemap(): MetadataRoute.Sitemap {
  const origin =
    process.env.NEXT_PUBLIC_SITE_URL ||
    (process.env.VERCEL_PROJECT_PRODUCTION_URL
      ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
      : "http://localhost:4173");
  return ["/", "/explorer/", "/ablations/", "/about/"].map((p) => ({
    url: origin + p,
  }));
}
