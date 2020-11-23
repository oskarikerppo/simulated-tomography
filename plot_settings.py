import matplotlib.pyplot as plt


class PlotSettings:

	def __init__(self):
		#Bar with in rectangle plot
		self.bar_width = 0.35
		#Opacity used in rectangle plot
		self.opacity = 0.8
		#Size of points in scatter
		self.scatter_size = 0.2
		#Number of bins in histogram
		self.num_of_bins = 20
		#Width of scientific paper in PTS
		self.width = 360
		#Font settings for latex
		self.tex_fonts = {
		    # Use LaTeX to write all text
		    "text.usetex": True,
		    "font.family": "serif",
		    # Use 10pt font in plots, to match 10pt font in document
		    "axes.labelsize": 10,
		    "font.size": 10,
		    # Make the legend/label fonts a little smaller
		    "legend.fontsize": 8,
		    "xtick.labelsize": 8,
		    "ytick.labelsize": 8
		}

	#Transform width in pixels to width in inches.
	#Use golden ratio as ratio between width and height
	def set_size(self, fraction=1):
	    """Set figure dimensions to avoid scaling in LaTeX.

	    Parameters
	    ----------
	    width: float
	            Document textwidth or columnwidth in pts
	    fraction: float, optional
	            Fraction of the width which you wish the figure to occupy

	    Returns
	    -------
	    fig_dim: tuple
	            Dimensions of figure in inches
	    """
	    # Width of figure (in pts)
	    fig_width_pt = self.width * fraction

	    # Convert from pt to inches
	    inches_per_pt = 1 / 72.27

	    # Golden ratio to set aesthetic figure height
	    # https://disq.us/p/2940ij3
	    golden_ratio = (5**.5 - 1) / 2

	    # Figure width in inches
	    fig_width_in = fig_width_pt * inches_per_pt
	    # Figure height in inches
	    fig_height_in = fig_width_in * golden_ratio

	    fig_dim = (fig_width_in, fig_height_in)

	    return fig_dim







