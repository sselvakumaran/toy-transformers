# loss plotter, shows approximate global shape plus local values
# uses triangular binning (summary ~O(sqrt(n)))
# blits global summary unless there's a new bin

import matplotlib.pyplot as plt
import seaborn as sns
import bisect
from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple

class _PointHandler:
	def __init__(self):
		self.xs: List[float] = []
		self.ys: List[float] = []

	def __bool__(self):
		return len(self.xs) > 0 and len(self.ys) > 0
	
	def __len__(self):
		return len(self.xs)
	
	def add_point(self, 
		x: Optional[float] = None, 
		y: Optional[float] = None
	) -> None:
		if bool(self) and (x is not None and self.xs[-1] >= x):
			raise ValueError(f"x must be inputted in nondecreasing manner: {self.xs[-1]} >= {x}")
		if y is None:
			return # handle silently
		
		x = x if x is not None else self.get_last_x() + 1
		self.xs.append(x)
		self.ys.append(y)
	
	def add_points(self,
		xs: List[float],
		ys: List[float]
	):
		if len(xs) != len(ys):
			raise ValueError("lists are not of equal length")
		for x, y in zip(xs, ys):
			self.add_point(x, y)

	def get_last_point(self) -> Tuple[float, float]:
		if not self:
			raise ValueError("lists are not of equal length")
		return (self.xs[-1], self.ys[-1])
	
	def get_points(self):
		return (self.xs, self.ys)
	
	def get_last_x(self):
		return self.xs[-1] if self else 0
	
	def get_last_points(self,
		n: int = 50,
		by: Literal['x', 'index'] = 'index',
		include_previous: bool = True
	):
		if not self:
			return ([], [])

		match by:
			case 'index':
				start_idx = n + (1 if include_previous else 0)
				return (self.xs[-start_idx:], self.ys[-start_idx:])
			case 'x':
				start_val = self.get_last_x() - (n - 1)
				start_idx = bisect.bisect_left(self.xs, start_val) - (1 if include_previous else 0)
				return (self.xs[start_idx:], self.ys[start_idx:])
			case _:
				raise ValueError("invalid 'by' parameter")

	def downsample(self, include_latest: bool = False):
		out_xs = []
		out_ys = []
		s1, s2 = 0, 0
		while s2 < len(self):
			out_xs.append(self.xs[s2])
			out_ys.append(self.ys[s2])
			s1 += 1; s2 += s1
		
		if include_latest and s2 + 1 != len(self):
			out_xs.append(self.xs[-1])
			out_ys.append(self.ys[-1])
		
		return (out_xs, out_ys)

class LossTracker:

	@dataclass()
	class _Cache:
		t_bins: int = 0
		v_bins: int = 0
		t_line = None
		v_line = None
		background = None
		ax1 = None
		ax2 = None

	def __init__(self, 
		style: str = "ticks", 
		context: str = "notebook",
		palette: str = "colorblind"
	):
		sns.set_style(style)
		sns.set_context(context, font_scale=1, rc={"lines.linewidth": 2})
		sns.set_palette(palette)
		self.fig = plt.figure(figsize=(8, 5))

		self.train_points = _PointHandler()
		self.val_points = _PointHandler()
		
		self._cache = self._Cache()

	def add_points(self,
		train: float, 
		val: Optional[float] = None, 
		x: Optional[int] = None
	) -> None:
		x = x if x is not None else self.train_points.get_last_x() + 1
		self.train_points.add_point(x=x, y=train)
		self.val_points.add_point(x=x, y=val)

	def add_points_batch(self,
		trains: List[float],
		vals: List[Optional[float]] = None,
		start_x: Optional[int] = None
	) -> None:
		start_x = start_x if start_x is not None else self.train_points.get_last_x() + 1
		xs = list(range(start_x, start_x + len(trains)))
		self.train_points.add_points(xs, trains)
		self.val_points.add_points(xs, vals)

	def _setup_axes(self):
		self.fig.clear()
		gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.1)
		ax1 = self.fig.add_subplot(gs[0, 0])
		ax2 = self.fig.add_subplot(gs[0, 1], sharey=ax1)

		sns.despine(ax=ax1, offset=0.2)
		sns.despine(ax=ax2, left=True, offset=0.2)
		ax2.get_yaxis().set_visible(False)

		self._cache.ax1, self._cache.ax2 = ax1, ax2
		return ax1, ax2

	def render_fast_figure(self) -> plt.Figure:
		ax1, ax2 = self._cache.ax1, self._cache.ax2

		if ax1 is None or ax2 is None:
			ax1, ax2 = self._setup_axes()

		t_xs, t_ys = self.train_points.downsample()
		v_xs, v_ys = self.val_points.downsample()

		t_bins, v_bins = len(t_xs), len(v_xs)
		ax1_needs_update = (
			t_bins != self._cache.t_bins or
			v_bins != self._cache.v_bins
		)

		if ax1_needs_update:
			ax1.clear()
			sns.despine(ax=ax1, offset=0.2)
			if t_bins > 0:
				t_line, = ax1.plot(t_xs, t_ys, alpha=0.6, label="train")
			else:
				t_line = None
			
			if v_bins > 0:
				v_line, = ax1.plot(v_xs, v_ys, label="val")
			else:
				v_line = None
			
			self._cache.t_bins = t_bins
			self._cache.v_bins = v_bins
			self._cache.t_line = t_line
			self._cache.v_line = v_line
			self._cache.background = self.fig.canvas.copy_from_bbox(ax1.bbox)

			handles, labels = ax1.get_legend_handles_labels()
			if handles:
				if ax1.get_legend() is not None:
					ax1.get_legend().remove()
				self.fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9))
		else:
			self.fig.canvas.restore_region(self._cache.background)
			for line in [self._cache.t_line, self._cache.v_line]:
				if line is not None:
					ax1.draw_artist(line)
			self.fig.canvas.blit(ax1.bbox)

		# right axes: plot local loss
		ax2.clear()
		t_xs, t_ys = self.train_points.get_last_points()
		v_xs, v_ys = self.val_points.get_last_points(by='x')
		if t_xs:
			ax2.plot(t_xs, t_ys, alpha=0.6)
		if v_xs:
			ax2.plot(v_xs, v_ys)
		if len(self.train_points) > 50:
			ax2.set_xlim(left=t_xs[1])
		sns.despine(ax=ax2, left=True, offset=0.2)
		ax2.get_yaxis().set_visible(False)
		self.fig.canvas.draw_idle()
		return self.fig
	
	def render_full_figure(self) -> plt.Figure:
		pass
