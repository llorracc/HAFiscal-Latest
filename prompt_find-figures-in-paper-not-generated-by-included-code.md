# Task: Identify Figures Referenced in LaTeX Files Without Creation Code

## Objective
Find and list all figures referenced in LaTeX files that don't have corresponding creation code in Python files or Jupyter notebooks.

## Steps

1. Find all .tex files in the project root directory and in the Subfiles directory and in the Figures directory
- Exclude instances of \includegraphics on lines that begin with a %
- Exclude instances of includegraphics whose pathname includes 'private'
    - for example, any files in Presentations_private/

2. For each figure filename:
   - Remove any path prefixes (e.g., "Code/HA-Models/FromPandemicCode/Figures/")
   - Remove any file extensions (e.g., ".pdf", ".png")
   - Remove any LaTeX-specific formatting (e.g., \FigDir/)
   - Create a clean list of unique figure base names

3. For each figure base name:
   a. First, search for files that might contain the creation code:
      - Look for Python files (.py) and Jupyter notebooks (.ipynb) with names containing key parts of the figure name (e.g., for "IMPCs_wSplEstimated", search for files containing "IMPC" or "Spl")
      - Check the directory where the figure is referenced to find nearby Python files and notebooks
   
   b. Then, within those files, search for the figure name in:
      - Direct filename references: "figure_name"
      - Variable names: figure_name
      - Function calls: plot_figure_name(), make_figs('figure_name')
      - Save commands: savefig(), show_plot(save_path=)
      - String literals containing the name
      - Comments that might indicate creation code

## Search Strategy
1. Start with a broad search for the figure name across all Python files (.py) and Jupyter notebooks (.ipynb)
2. For each match, examine the surrounding context to determine if it's actually creating the figure
3. If no matches found, look for files with similar names that might contain the creation code
4. Check the directory structure to find related files that might contain the creation code
5. Look for commented-out code that might have created the figure
6. Consider variations in naming conventions (e.g., underscores vs. hyphens)

## Output Format
1. List of figures with no executable creation code found, including:
   - Original filename
   - Cleaned base name
   - Files searched
   - Any partial matches found
   
2. List of figures which may have been created by code now commented out
   - Original filename
   - Cleaned base name
   - Files searched
   - Any partial matches found

## Notes
- Do not report back on figures whose creation code was found
- Consider case sensitivity in searches
- Account for different naming conventions (e.g., underscores vs. hyphens)
- Check for commented-out code that might create these figures
- Look for similar names that might indicate the same figure
- Check files in the same directory as the figure reference first
- Look for files with names that suggest they might create the figure (e.g., CreateIMPCfig.py for IMPC-related figures)

For all of the files that do not have clear creation codes in python files, search to see if you can find code in .ipynb files that creates them
