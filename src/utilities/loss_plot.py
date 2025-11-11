# NEW VERSION

import matplotlib.pyplot as plt
import seaborn as sns
import bisect
import math
from typing import Optional, List, Literal, Tuple
from dataclasses import dataclass

# store 2d points easier
# assumes inputted in non-decreasing order
# handles exponential decay 
class _PointHandler:
	def __init__(self,
		downsample_mode: Literal['exponential', 'triangular'] = 'triangular'
	):
		self.xs = []
		self.ys = []

		self.downsample_mode = downsample_mode
		if self.downsample_mode == "exponential":
			self.smoothed_ys = []
	
	def __bool__(self):
		return len(self.xs) > 0 and len(self.ys) > 0
	
	def __len__(self):
		return len(self.xs)

	def add_point(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
		if y is None:
			return # convenient to put in optional arguments without case
		x = x if x is not None else self.get_last_x() + 1
		self.xs.append(x)
		self.ys.append(y)

		if self.downsample_mode == "exponential":
			self.smoothed_ys.append(
				self.alpha * y + (1 - self.alpha) * self.smoothed_ys[-1]
				if len(self.smoothed_ys) > 0 else y
			)
	
	def add_points(self, xs: List[int], ys: List[float]) -> None:
		if len(xs) != len(ys):
			raise ValueError("lists are not of equal length")
		for x, y in zip(xs, ys):
			self.add_point(x, y)	
	
	def get_last_point(self) -> Tuple[int, float]:
		if not self:
			return (0, -1)
		return (self.xs[-1], self.ys[-1])
	
	def get_points(self):
		return (self.xs, self.ys)
	
	def get_last_x(self):
		return self.xs[-1] if self else 0
	
	def get_last_points(self, 
		n: int = 50, 
		by: Literal['x', 'index'] = 'index',
		include_previous: bool = True # include one more by index
	):
		if not self:
			return ([], [])
		
		if by == 'index':
			start_idx = n + (1 if include_previous else 0)
			return(self.xs[-start_idx:], self.ys[-start_idx:])
		
		start_val = self.get_last_x() - (n - 1)
		start_idx = bisect.bisect_left(self.xs, start_val) - (1 if include_previous else 0)

		return (self.xs[start_idx:], self.ys[start_idx:])
	
	def downsample(self):
		match self.downsample_mode:
			case "exponential":
				return self._downsample_exponential()
			case "triangular":
				return self._downsample_triangular()
			case _:
				raise ValueError("downsample method selected does not exist")
		
	def _downsample_exponential(self, 
		resolution: int = 50
	):
		N = len(self.smoothed_ys)
		if N <= resolution:
			return (self.xs, self.smoothed_ys)
		start_x, end_x = self.xs[0], self.xs[-1]
		bin_size = (end_x - start_x) / (resolution - 1)
		idx_per_bin = (N - 1) / (resolution - 1)
		bin_xs = []
		bin_ys = []
		for i in range(resolution):
			ideal_idx = i * idx_per_bin
			l = int(ideal_idx)
			t = ideal_idx - l
			y0 = self.smoothed_ys[l]
			y1 = self.smoothed_ys[l + 1] if l + 1 < N else y0

			bin_xs.append(start_x + i * bin_size)
			bin_ys.append(y0 * (1 - t) + y1 * t)
		return (bin_xs, bin_ys)
	
	def _downsample_triangular(self):
		out_xs = []
		out_ys = []
		s1, s2 = 0, 0
		while s2 < len(self):
			out_xs.append(self.xs[s2])
			out_ys.append(self.ys[s2])
			s1 += 1; s2 += s1
		
		if s2 + 1 != len(self):
			out_xs.append(self.xs[-1])
			out_ys.append(self.ys[-1])

		return (out_xs, out_ys)
			

# TODO: fix
class LossTracker:
	# initialize settings
	def __init__(self,
			palette: str = "colorblind"
		):
		sns.set_style("ticks")
		sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
		sns.set_palette(palette)
		self.fig = plt.figure(figsize=(8, 5))

		self.train_points = _PointHandler()
		self.val_points = _PointHandler()

	def add_points(self, 
		train: float, 
		val: Optional[float] = None, 
		x: Optional[int] = None
	):
		x = x if x is not None else self.train_points.get_last_x() + 1
		self.train_points.add_point(x=x, y=train)
		self.val_points.add_point(x=x, y=val)

	def add_points_batch(self, 
		trains: List[float], 
		vals: List[Optional[float]] = None,
		start_x : Optional[int] = None
	):
		start_x = start_x if start_x is not None else self.train_points.get_last_x() + 1
		xs = list(range(start_x, start_x + len(trains)))
		self.train_points.add_points(xs, trains)
		self.val_points.add_points(xs, vals)

	def render_fast_figure(self) -> plt.Figure:
		self.fig.clear()

		gs = self.fig.add_gridspec(1, 2, width_ratios=[3,1], wspace=0.1)
		ax1 = self.fig.add_subplot(gs[0, 0])
		ax2 = self.fig.add_subplot(gs[0, 1], sharey=ax1)

		# plot 1 lines
		if self.train_points:
			xs, ys = self.train_points.downsample()
			sns.lineplot(x=xs, y=ys, ax=ax1, alpha=0.6, label="train")
		if self.val_points:
			xs, ys = self.val_points.downsample()
			sns.lineplot(x=xs, y=ys, ax=ax1, label="val")

		# plot 1 last values
		if self.train_points:
			last_train_point = self.train_points.get_last_point()
			ax1.scatter([last_train_point[0]], [last_train_point[1]], s=40, zorder=3)
		if self.val_points:
			last_val_point = self.val_points.get_last_point()
			ax1.scatter([last_val_point[0]], [last_val_point[1]], s=40, zorder=3)
		
		# plot 2 local loss plot
		recent_train_points = self.train_points.get_last_points()
		recent_val_points = self.val_points.get_last_points(by='x')
		sns.lineplot(x=recent_train_points[0], y=recent_train_points[1], ax=ax2, alpha=0.6)
		sns.lineplot(x=recent_val_points[0], y=recent_val_points[1], ax=ax2)

		ax2.get_yaxis().set_visible(False)
		if len(self.train_points) > 50:
			ax2.set_xlim(left=recent_train_points[0][1])

		sns.despine(ax=ax1, offset=0.2)
		sns.despine(ax=ax2, left = True, offset=0.2)

		handles, labels = ax1.get_legend_handles_labels()
		if handles:
			if ax1.get_legend() is not None:
				ax1.get_legend().remove()
			self.fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9))

		return self.fig

	# Render and return figure to print
	def render_full_figure(self) -> plt.Figure:
		pass