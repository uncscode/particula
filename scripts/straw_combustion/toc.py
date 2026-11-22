"""Redraw the straw study card from its archived joint fit."""

from scripts.straw_combustion.plot_figures import draw_toc, selected_rows

if __name__ == "__main__":
    draw_toc(selected_rows())
