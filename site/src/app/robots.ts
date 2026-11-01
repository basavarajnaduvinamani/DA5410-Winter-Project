import type { MetadataRoute } from "next";
export const dynamic = "force-static";
export default function robots(): MetadataRoute.Robots {
  const origin =
    process.env.NEXT_PUBLIC_SITE_URL ||
    (process.env.VERCEL_PROJECT_PRODUCTION_URL
      ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
      : "http://localhost:4173");
  return {
    rules: { userAgent: "*", allow: "/" },
    sitemap: `${origin}/sitemap.xml`,
  };
}
