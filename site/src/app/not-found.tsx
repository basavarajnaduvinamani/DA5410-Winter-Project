import Link from "next/link";
export default function NotFound() {
  return (
    <div className="not-found narrow-page">
      <p className="eyebrow">PAGE NOT FOUND</p>
      <h1>This path has no evidence.</h1>
      <p>
        The page may have moved, or the address may be incomplete. The research
        record and example explorer are still available.
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
