import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { Header, Footer } from "@/components/shell";
import "./globals.css";
const origin =
  process.env.NEXT_PUBLIC_SITE_URL ||
  (process.env.VERCEL_PROJECT_PRODUCTION_URL
    ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
    : "https://counterprior.vercel.app");
export const metadata: Metadata = {
  metadataBase: new URL(origin),
  title: {
    default: "CounterPrior | Do multimodal models know when they are guessing?",
    template: "%s — CounterPrior",
  },
  description:
    "Object and sound grounding in LLaVA, Gemma 4 E2B and Qwen2.5-Omni-3B, with held-out results and recorded limitations. LLaVA work at AI4Bharat, IIT Madras (Prof. Mitesh Khapra), Nov 2025 to Feb 2026. Extended to audio and video in Nov 2026.",
  openGraph: {
    title: "CounterPrior | Do multimodal models know when they are guessing?",
    description:
      "Question, frozen protocol, honest results and reproducibility for representation-level hallucination research.",
    type: "website",
    images: [
      {
        url: "/opengraph.png",
        width: 1200,
        height: 630,
        alt: "The CounterPrior research page",
      },
    ],
  },
  twitter: { card: "summary_large_image" },
  icons: { icon: "/favicon.svg" },
  robots: { index: true, follow: true },
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={GeistSans.variable}
    >
      <body>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <Header />
        <main id="main">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
