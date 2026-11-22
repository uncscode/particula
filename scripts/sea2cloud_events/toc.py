"""Render the Sea2Cloud study card from the saved arrival-time comparison."""

import pandas as pd
from scripts.render_literature_tocs import ROOT, dots_svg


def graphical_abstract(lead_data):
    """Show the first held-out spectrum after matching the 06:00 size count."""
    next_bin = lead_data[
        (lead_data.tau_h == 2)
        & (lead_data.vapor_million == 1)
        & (lead_data.hour == 78.5)
    ].set_index("lead_h")
    observed = next_bin.loc[0, "observed"]
    same_time = next_bin.loc[0, "predicted"]
    earlier = next_bin.loc[1, "predicted"]
    scale = 195 / max(observed, same_time, earlier)
    graphic = f'''<svg xmlns="http://www.w3.org/2000/svg" width="3in"
  height="3in" viewBox="0 0 300 300" role="img" aria-labelledby="title desc">
  <title id="title">Sea2Cloud ship measurement and arrival scenarios</title>
  <desc id="desc">At 06:30 UTC on March 22, Tangaroa measured {observed:.0f}
  particles/cm³ in the 12–25 nm band. The two-hour exchange model predicts
  {same_time:.0f} particles/cm³ when the incoming 16 nm particles arrive with
  the small-particle pulse and {earlier:.0f} when they arrive one hour before
  it. Each model used the 06:00 spectrum for calibration.</desc>
  <rect width="300" height="300" fill="#fbf2e3"/>
  <g fill="#243e43">
    <path d="M21 39h61l-9 10H31z M36 30h15v9H36z M54 24h10v15H54z
      M57 16h3v8h-3z"/>
    <path d="M20 56q8 4 16 0t16 0t16 0t16 0" fill="none"
      stroke="#617679" stroke-width="2.5"/>
  </g>
  <g fill="#243e43" font-family="sans-serif">
    <text x="96" y="32" font-size="19"
      font-weight="bold">12–25 nm particles</text>
    <text x="96" y="55" font-size="15">per cm³ · 06:30 UTC</text>
    <text x="22" y="94" font-size="17"
      font-weight="bold">Ship measurement</text>
    <rect x="22" y="106" width="{observed * scale:.2f}"
      height="13" fill="#243e43"/>
    <text x="{30 + observed * scale:.2f}" y="118"
      font-size="17">{observed:.0f}</text>
    <path d="M22 137h256" stroke="#617679" stroke-width="1"/>
    <text x="22" y="162" font-size="17"
      font-weight="bold">Modeled 16 nm arrival</text>
    <text x="22" y="189" font-size="16">With the small-particle pulse</text>
    <rect x="22" y="200" width="{same_time * scale:.2f}"
      height="13" fill="#a44d21"/>
    <text x="{30 + same_time * scale:.2f}" y="212"
      font-size="17">{same_time:.0f}</text>
    <text x="22" y="246" font-size="16">1 h before the pulse</text>
    <rect x="22" y="257" width="{earlier * scale:.2f}"
      height="13" fill="#135963"/>
    <text x="{30 + earlier * scale:.2f}" y="269"
      font-size="17">{earlier:.0f}</text>
  </g>
<g transform="translate(264 270)">{dots_svg()}</g>
</svg>
'''
    (ROOT / "docs/images/sea2cloud-events/graphical-abstract.svg").write_text(
        graphic
    )


if __name__ == "__main__":
    graphical_abstract(
        pd.read_csv(
            ROOT / "website/public/study-data/sea2cloud/incoming-size-lead.csv"
        )
    )
