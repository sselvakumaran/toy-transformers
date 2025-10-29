# NEW VERSION

import matplotlib.pyplot as plt
import seaborn as sns
import bisect
from itertools import zip_longest
from typing import Optional, List, Literal, Tuple

# store 2d points easier
# assumes inputted in non-decreasing order
class _PointHandler:
	def __init__(self):
		self.xs = []
		self.ys = []
	
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
	
	def add_points(self, xs: List[int], ys: List[int]) -> None:
		if len(xs) != len(ys):
			raise ValueError("lists are not of equal length")

		xs_filtered, ys_filtered = zip(
			*filter(
				lambda t: t[1] is not None, 
				zip(xs, ys)
		))
		
		self.xs.extend(xs_filtered)
		self.ys.extend(ys_filtered)
	
	def get_last_point(self) -> Tuple[int, int]:
		if not self:
			return 0
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

		trains = self.train_points.get_points()
		vals = self.val_points.get_points()

		# plot 1
		sns.lineplot(x=trains[0], y=trains[1], ax=ax1, alpha=0.6, label="train")
		sns.lineplot(x=vals[0], y=vals[1], ax=ax1, label="val")
		

		if self.train_points:
			last_train_point = self.train_points.get_last_point()
			ax1.scatter([last_train_point[0]], [last_train_point[1]], s=40, zorder=3)
		if self.val_points:
			last_val_point = self.val_points.get_last_point()
			ax1.scatter([last_val_point[0]], [last_val_point[1]], s=40, zorder=3)
		
		# plot 2
		recent_train_points = self.train_points.get_last_points()
		recent_val_points = self.val_points.get_last_points(by='x')
		sns.lineplot(x=recent_train_points[0], y=recent_train_points[1], ax=ax2, alpha=0.6)
		sns.lineplot(x=recent_val_points[0], y=recent_val_points[1], ax=ax2)

		ax2.get_yaxis().set_visible(False)

		if len(self.train_points) > 50:
			print(f"{len(self.train_points) - 49} {recent_val_points[0]}")
			ax2.set_xlim(left=recent_train_points[0][1])

		handles, labels = ax1.get_legend_handles_labels()
		if handles:
			if ax1.get_legend() is not None:
				ax1.get_legend().remove()
			self.fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9))

		sns.despine(ax=ax1, offset=0.2)
		sns.despine(ax=ax2, left = True, offset=0.2)

		return self.fig

	# Render and return figure to print
	def render_full_figure(self) -> plt.Figure:
		pass
		# if self.y_train:
		# 	if self.train_line:
		# 		self.train_line.set_data(self.x, self.y_train)
		# 	else:
		# 		sns.lineplot(x=self.x, y=self.y_train, ax=self.ax1)
		# 		self.train_line = self.ax1.lines[0]

		# if any(y is not None for y in self.y_val):
		# 	if self.val_line:
		# 		self.val_line.set_data(self.x, self.y_val)
		# 	else:
		# 		sns.lineplot(x=self.x, y=self.y_val, ax=self.ax1)
		# 		self.val_line = self.ax1.lines[1]
		
		# self.ax1.relim()
		# self.ax1.autoscale_view()

		# self.fig.canvas.draw_idle()
		# return self.fig