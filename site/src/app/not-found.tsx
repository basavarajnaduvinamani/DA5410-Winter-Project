import Link from "next/link";
export default function NotFound() {
  return (
    <div className="not-found narrow-page">
      <p className="eyebrow">Page not found</p>
      <h1>This page does not exist.</h1>
      <p>
        The page may have moved, or the address may be incomplete. Start from the
        results or the example explorer.
      </p>
      <div className="resource-links">
        <Link className="button-primary" href="/">
          Return to the research
        </Link>
        <Link href="/explorer/">Open the explorer</Link>
      </div>
    </div>
  );
}
