# Set interaction mode based on environment variable
$pdflatex = 'pdflatex -interaction=nonstopmode' if $ENV{'LATEX_INTERACTION_MODE'} eq 'noninteractive';
