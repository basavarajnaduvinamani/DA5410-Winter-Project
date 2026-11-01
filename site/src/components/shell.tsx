import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";
import { code, sourceUrl } from "@/lib/data";
export function Header() {
  return (
    <header className="site-header">
      <div className="header-inner">
        <Link className="wordmark" href="/">
          <span className="monogram">CP</span>
          <span>CounterPrior</span>
        </Link>
        <nav aria-label="Main navigation">
          <Link href="/#results">Research</Link>
          <Link href="/explorer/">Explorer</Link>
          <Link href="/ablations/">Ablations</Link>
          <Link href="/about/">About</Link>
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}
export function Footer() {
  return (
    <footer className="site-footer">
      <div>
        <strong>CounterPrior</strong>
        <p>A research record. Evidence before claims.</p>
      </div>
      <div className="footer-links">
        <a href={code}>
          Source code <span aria-hidden="true">↗</span>
        </a>
        <a href="/research.json" download>
          Evidence JSON
        </a>
        <Link href="/about/#licensing">Licensing & privacy</Link>
        <a href={sourceUrl("docs/PROTOCOL.md")}>Protocol</a>
      </div>
    </footer>
  );
}
export function Source({ path, label }: { path: string; label?: string }) {
  return (
    <a className="source" href={sourceUrl(path)}>
      {label || `Source: ${path}`} <span aria-hidden="true">↗</span>
    </a>
  );
}
export function PendingNote({
  title = "Awaiting verified results",
  children,
}: {
  title?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="evidence-note" role="note">
      <span className="status-mark" aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        {children ? <div className="note-body">{children}</div> : null}
      </div>
    </div>
  );
}
export function PageIntro({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="page-intro">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <div className="intro-copy">{children}</div>
    </div>
  );
}
