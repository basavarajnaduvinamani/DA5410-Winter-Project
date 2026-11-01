export function PipelineFigure() {
  return (
    <figure className="method-figure">
      <div className="figure-top">
        <span className="figure-label">
          METHOD / REPRESENTATION → INTERVENTION
        </span>
        <span className="figure-label">Conceptual diagram</span>
      </div>
      <svg
        className="pipeline-svg"
        viewBox="0 0 760 292"
        role="img"
        aria-labelledby="pipeline-title pipeline-desc"
      >
        <title id="pipeline-title">CounterPrior method</title>
        <desc id="pipeline-desc">
          An image and question enter a vision-language model. A probe reads the
          terminal prompt hidden state. A separately calibrated cosine gate
          determines whether SLERP rotates the activation before yes or no
          generation. Layer and gate are chosen on calibration; strength is
          chosen on validation.
        </desc>
        <defs>
          <marker
            id="arrow"
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <path d="M0 0L7 3.5L0 7" fill="var(--muted)" />
          </marker>
        </defs>
        <g fill="none" stroke="var(--line)" strokeWidth="1.5">
          <rect x="24" y="107" width="126" height="72" rx="6" />
          <rect x="202" y="80" width="172" height="126" rx="6" />
          <path d="M150 143h49M374 143h48M558 143h48" markerEnd="url(#arrow)" />
          <rect x="428" y="107" width="130" height="72" rx="6" />
          <rect x="612" y="107" width="124" height="72" rx="6" />
          <path d="M249 101v83m22-83v83m22-83v83m22-83v83m22-83v83" />
          <path
            d="M290 80V48h202v56"
            stroke="var(--accent)"
            markerEnd="url(#arrow)"
          />
        </g>
        <g fill="var(--ink)" fontSize="14" textAnchor="middle">
          <text x="87" y="137">
            Image
          </text>
          <text x="87" y="158">
            + question
          </text>
          <text x="288" y="228">
            Decoder layers
          </text>
          <text x="493" y="137">
            Gate + SLERP
          </text>
          <text x="493" y="158" fill="var(--muted)" fontSize="12">
            conditional rotation
          </text>
          <text x="674" y="137">
            Answer
          </text>
          <text x="674" y="158" fill="var(--muted)" fontSize="12">
            yes / no
          </text>
          <text x="391" y="35" fill="var(--accent)">
            Terminal-state probe
          </text>
          <text x="288" y="258" fill="var(--muted)" fontSize="12">
            calibration: layer + direction + gate
          </text>
          <text x="550" y="258" fill="var(--muted)" fontSize="12">
            validation: strength
          </text>
        </g>
        <circle cx="290" cy="80" r="4" fill="var(--accent)" />
      </svg>
      <figcaption>
        A readout before answer generation; a controlled intervention during
        decoding. The answer-token confidence baselines are available only after
        the first output token.
      </figcaption>
    </figure>
  );
}
export function GeometryFigure() {
  return (
    <figure className="geometry-figure">
      <svg
        viewBox="0 0 380 280"
        role="img"
        aria-labelledby="geometry-title geometry-desc"
      >
        <title id="geometry-title">
          SLERP and vector addition on the activation sphere
        </title>
        <desc id="geometry-desc">
          Spherical interpolation follows a circular arc at constant activation
          norm. Vector addition follows a straight path that can change the
          norm. This is a geometric illustration, not an observed model result.
        </desc>
        <defs>
          <marker
            id="geometry-arrow"
            markerWidth="6"
            markerHeight="6"
            refX="5"
            refY="3"
            orient="auto"
          >
            <path d="M0 0L6 3L0 6" fill="var(--ink)" />
          </marker>
        </defs>
        <circle
          cx="161"
          cy="158"
          r="88"
          fill="none"
          stroke="var(--line)"
          strokeDasharray="4 5"
        />
        <path d="M48 158h238M161 266V40" stroke="var(--line)" />
        <path
          d="M161 158l68-56"
          stroke="var(--ink)"
          strokeWidth="2"
          markerEnd="url(#geometry-arrow)"
        />
        <path d="M161 158l-23-85" stroke="var(--accent)" strokeWidth="2" />
        <path
          d="M229 102A88 88 0 0 0 138 73"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="3"
        />
        <path
          d="M229 102l-15-61"
          stroke="var(--ink)"
          strokeDasharray="4 4"
          strokeWidth="2"
          markerEnd="url(#geometry-arrow)"
        />
        <circle cx="229" cy="102" r="4" fill="var(--ink)" />
        <circle cx="138" cy="73" r="4" fill="var(--accent)" />
        <g fill="var(--ink)" fontSize="13">
          <text x="234" y="118">
            h
          </text>
          <text x="219" y="36">
            h + βv
          </text>
          <text x="72" y="60" fill="var(--accent)">
            SLERP(h, v, α)
          </text>
          <text x="277" y="179" fill="var(--muted)">
            ‖h‖
          </text>
        </g>
      </svg>
      <figcaption>
        <strong>Same norm, different direction.</strong> SLERP rotates along the
        activation sphere. Addition can change its radius. A preserved norm
        alone does not prove factual improvement.
      </figcaption>
    </figure>
  );
}
