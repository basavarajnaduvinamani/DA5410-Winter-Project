import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Header, Footer } from "@/components/shell";
import "./globals.css";
const origin =
  process.env.NEXT_PUBLIC_SITE_URL ||
  (process.env.VERCEL_PROJECT_PRODUCTION_URL
    ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
    : "http://localhost:4173");
export const metadata: Metadata = {
  metadataBase: new URL(origin),
  title: {
    default: "CounterPrior — Evidence before claims",
    template: "%s — CounterPrior",
  },
  description:
    "Research at AI4Bharat Lab, IIT Madras: detecting answer errors and testing norm-preserving steering in LLaVA on image-disjoint POPE.",
  openGraph: {
    title: "CounterPrior — Evidence before claims",
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
const themeScript = `try { var t = localStorage.getItem('counterprior-theme'); document.documentElement.dataset.theme = t === 'dark' || t === 'light' ? t : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch (_) {}`;
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${GeistSans.variable} ${GeistMono.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
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
