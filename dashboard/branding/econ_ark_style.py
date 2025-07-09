"""
Econ-ARK branding and style definitions for use in Python applications.
Contains color schemes, plot styles, and HTML/CSS styling.
"""

from cycler import cycler

# Define Econ-ARK brand colors and styling
ARK_BLUE = "#005b8f"      # primary
ARK_LIGHTBLUE = "#0ea5e9" # lighter accent
ARK_ORANGE = "#f97316"    # accent
ARK_GREEN = "#047857"     # accent
ARK_SLATE_DK = "#1e293b"  # dark text
ARK_SLATE_LT = "#475569"  # light text
ARK_GREY = "#6b7280"      # utility
ARK_GRID = "#e2e8f0"      # grid lines
ARK_PANEL = "#f1f5f9"     # panel background

# Define refined Econ-ARK colors and styling
ARK_PANEL_LIGHT = "#f8fafc"  # Lighter panel background
ARK_GRID_SOFT = "#edf2f7"    # Softer grid lines
ARK_SPINE = "#94a3b8"        # Professional spine color
ARK_TEXT = "#334155"         # Clear, professional text color

# Matplotlib style configuration
MATPLOTLIB_STYLE = {
    # --- Font & text ---
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "600",  # Bolder titles
    "axes.labelsize": 9,  # Smaller axis labels
    "axes.labelweight": "500",  # Slightly bolder labels
    "xtick.labelsize": 8.5,  # Slightly smaller tick labels
    "ytick.labelsize": 8.5,
    # Text colors
    "text.color": ARK_TEXT,
    "axes.labelcolor": ARK_TEXT,
    "axes.titlecolor": ARK_BLUE,  # Brand color for all titles including subplots
    "xtick.color": ARK_TEXT,
    "ytick.color": ARK_TEXT,
    
    # --- Colours & lines ---
    "axes.prop_cycle": cycler(color=[ARK_BLUE, ARK_ORANGE, ARK_GREEN, ARK_LIGHTBLUE, ARK_SLATE_LT]),
    "axes.edgecolor": ARK_SPINE,
    "axes.linewidth": 1.2,  # Slightly thicker spines
    "grid.color": ARK_GRID_SOFT,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,  # Subtle grid
    
    # --- Background & figure ---
    "axes.facecolor": ARK_PANEL_LIGHT,  # Very light blue-gray background
    "figure.facecolor": "white",
    "figure.dpi": 110,
    
    # --- Spines ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    
    # --- Legend ---
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": ARK_SPINE,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    
    # --- Lines & markers ---
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": "white",  # White edge on markers
    
    # --- Ticks ---
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
}

# HTML/CSS styles for dashboard
DASHBOARD_CSS = """
<style>
    :root {
        /* Brand colors */
        --ark-blue: #005b8f;          /* primary */
        --ark-lightblue: #0ea5e9;     /* lighter accent */
        --ark-slate-dk: #1e293b;      /* dark text */
        --ark-slate-lt: #475569;      /* light text */
        --ark-grid: #e2e8f0;          /* grid lines */
        --ark-panel: #f1f5f9;         /* panel background */
        --ark-body: 0.95rem;          /* base text size */
    }

    /* ===== HEADING SCALE & DECORATION ===== */
    h1, .ark-h1 {
        font:700 1.65rem/1.3 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
        color:var(--ark-blue);
    }

    h2, .ark-h2 {
        font:600 1.35rem/1.35 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
        color:var(--ark-slate-dk);
        position:relative;
        margin-bottom:1.1rem;
    }
    h2::after, .ark-h2::after {              /* full-width underline bar */
        content:'';
        position:absolute;
        left:0; bottom:-6px;
        width:100%; height:2px;
        background:var(--ark-blue);
        opacity: 0.9;
    }

    /* Optional lighter accent for figures */
    .ark-h2.lightblue { color:var(--ark-lightblue); }
    .ark-h2.lightblue::after { 
        background:var(--ark-lightblue);
        opacity: 0.8;  /* Slightly more transparent for lightblue */
    }

    h3, .ark-h3 {
        font:600 1.15rem/1.4 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
        color:var(--ark-slate-dk);
    }

    /* Body and utility classes */
    p.ark-body {font-size:var(--ark-body);line-height:1.45;color:var(--ark-slate-dk);}
    .ark-num {color:var(--ark-blue);font-weight:600;}
    .ark-label {font-size:0.9rem;font-weight:500;color:var(--ark-slate-lt);}
    
    /* Parameter group boxes */
    .param-group {
        background-color: var(--ark-panel);
        padding: 1.25em;
        margin: 0.75em 0 1.75em 0;
        border-radius: 8px;
        border: 1px solid var(--ark-grid);
    }
</style>
"""

# Header HTML with Econ-ARK logo
HEADER_HTML = """
<style>
  .ark-header {display:flex;align-items:center;
               background:#005b8f;padding:20px 24px;
               position:sticky;top:0;z-index:1000;
               box-shadow:0 2px 8px rgba(0,0,0,0.15);}
  .ark-header img {height:48px;margin-right:20px;}
  .ark-header span {color:#fff;font:600 1.35rem/1 system-ui,sans-serif;letter-spacing:-0.01em;}
</style>
<div class='ark-header'>
  <a href='https://econ-ark.org' target='_blank' style='border:0'>
    <img src='https://econ-ark.org/assets/img/econ-ark-logo-white.png'
         alt='Econ‑ARK logo'>
  </a>
  <span>HANK‑SAM Interactive Dashboard</span>
</div>
"""

def tidy_legend(fig):
    """Helper to format legends consistently across all figures."""
    # Remove any existing legends from subplots
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Get all unique handles and labels from all subplots
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:  # Only add unique items
                handles.append(handle)
                labels.append(label)
    
    # Add single legend below the figure
    fig.legend(handles, labels,
              bbox_to_anchor=(0.5, -0.1),  # Position below figure
              loc='center',
              ncol=3,  # Horizontal 3-column layout
              prop={'size': 9},
              frameon=True,
              framealpha=0.95,
              edgecolor='#cbd5e1')
    
    # Adjust subplot spacing
    fig.subplots_adjust(bottom=0.25, wspace=0.2) 