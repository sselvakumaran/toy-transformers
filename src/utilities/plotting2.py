import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output

class LossPlotter:
    def __init__(
        self,
        title: str = "loss",
        style: str = "darkgrid",
        figsize=(8, 5),
        smoothing: float = 0.0,    # 0 = no smoothing, e.g., 0.9 for EMA
        logy: bool = False,        # set to True for log-scale y-axis (loss must be > 0)
        update_every: int = 1,     # redraw only every N updates to reduce flicker
        ylim_padding: float = 0.05, # padding fraction for autoscale (now used for x-axis too)
        show_legend: bool = True
    ):
        sns.set_style(style)
        self.title = title
        self.smoothing = float(smoothing)
        self.logy = bool(logy)
        self.update_every = max(1, int(update_every))
        self.ylim_padding = float(ylim_padding)
        self.show_legend = show_legend

        # Data containers
        self.iterations = []
        self.train_losses_raw = []
        self.train_losses_display = []  # smoothed or raw for plotting
        self.val_iters = []
        self.val_losses = []

        self._ema = None  # for EMA smoothing
        # --- CHANGE 1: Removed "best" validation tracking ---
        # self._best_val = math.inf
        # self._best_val_iter = None

        # Matplotlib objects
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.train_line = None
        self.val_line = None
        # self.best_val_point = None # Removed
        self.latest_text = None
        # self.best_text = None # Renamed
        self.latest_val_text = None # --- CHANGE 2: Added latest_val_text ---

        self._setup_axes()
        plt.ion()

    # ---------------- Public API ---------------- #
    def update(self, train_loss: float, val_loss: float = None, iteration: int = None):
        """Add a data point and update the plot (throttled by update_every)."""
        iteration = int(iteration) if iteration is not None else len(self.train_losses_raw) + 1

        # Append raw train loss
        self.iterations.append(iteration)
        self.train_losses_raw.append(float(train_loss))

        # Update EMA if requested (displayed series)
        if self.smoothing > 0.0:
            if self._ema is None:
                self._ema = float(train_loss)
            else:
                # Note: This is an EMA (Exponential Moving Average)
                # alpha * new + (1 - alpha) * old
                # A high smoothing value (e.g., 0.9) means the new value has
                # *less* weight (alpha = 1 - 0.9 = 0.1).
                # Let's flip the logic to be more intuitive:
                # smoothing = 0.9 means 90% old, 10% new
                alpha = self.smoothing
                self._ema = alpha * self._ema + (1.0 - alpha) * float(train_loss)
            self.train_losses_display.append(self._ema)
        else:
            self.train_losses_display.append(float(train_loss))

        # Validation
        if val_loss is not None:
            val_loss = float(val_loss)
            self.val_iters.append(iteration)
            self.val_losses.append(val_loss)
            # --- CHANGE 1: Removed "best" validation tracking ---
            # if val_loss < self._best_val:
            #     self._best_val = val_loss
            #     self._best_val_iter = iteration

        # Redraw (throttled)
        if len(self.iterations) % self.update_every == 0 or val_loss is not None:
            self._draw_plot()

    def show(self):
        """Block to show the final figure (useful at end of training)."""
        plt.ioff()
        self._draw_plot(force=True)
        plt.show()

    def reset(self):
        """Clear history but keep the same figure/axes."""
        self.iterations.clear()
        self.train_losses_raw.clear()
        self.train_losses_display.clear()
        self.val_iters.clear()
        self.val_losses.clear()
        self._ema = None
        # --- CHANGE 1: Removed "best" validation tracking ---
        # self._best_val = math.inf
        # self._best_val_iter = None
        self._draw_plot(force=True)

    def save(self, path: str, dpi: int = 144, bbox_inches: str = "tight"):
        """Save the current plot to disk."""
        self._draw_plot(force=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)

    # ---------------- Internals ---------------- #
    def _setup_axes(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("loss")
        if self.logy:
            self.ax.set_yscale("log")

        # Initialize line artists
        (self.train_line,) = self.ax.plot([], [], label="train", color="tab:blue", lw=2)
        (self.val_line,) = self.ax.plot([], [], label="val", color="tab:orange", lw=2, marker='.', ms=8) # Added marker
        # --- CHANGE 1: Removed "best" validation point ---
        # (self.best_val_point,) = self.ax.plot([], [], "o", color="tab:orange", ms=6, alpha=0.9)

        if self.show_legend:
            self.ax.legend(loc="best")

    def _draw_plot(self, force: bool = False):
        # Update line data in place (fast)
        self.train_line.set_data(self.iterations, self.train_losses_display)
        
        # --- CHANGE 4: Simplified val_line update ---
        # No if/else needed, set_data handles empty lists
        self.val_line.set_data(self.val_iters, self.val_losses)
        
        # --- CHANGE 1: Removed "best" validation point ---
        # if len(self.val_iters) > 0:
        #     self.val_line.set_data(self.val_iters, self.val_losses)
        #     if self._best_val_iter is not None and math.isfinite(self._best_val):
        #         self.best_val_point.set_data([self._best_val_iter], [self._best_val])
        # else:
        #     self.val_line.set_data([], [])
        #     self.best_val_point.set_data([], [])

        # Annotations (latest train and latest val)
        self._update_annotations()

        # Autoscale with padding
        self._autoscale_with_padding()

        # Refresh notebook output
        clear_output(wait=True)
        display(self.fig)

    def _update_annotations(self):
        # Remove old annotations to avoid clutter
        if self.latest_text is not None:
            self.latest_text.remove()
            self.latest_text = None
        # --- CHANGE 2: Use latest_val_text ---
        if self.latest_val_text is not None:
            self.latest_val_text.remove()
            self.latest_val_text = None

        # Latest train label
        if self.iterations:
            x = self.iterations[-1]
            y_raw = self.train_losses_raw[-1]  # show raw numeric value
            y_disp = self.train_losses_display[-1] # anchor to displayed (smoothed) line
            self.latest_text = self.ax.annotate(
                f"{y_raw:.4f}",
                xy=(x, y_disp),
                xytext=(6, 0),
                textcoords="offset points",
                color="tab:blue",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )

        # --- CHANGE 3: Plot *latest* val text instead of *best* ---
        if self.val_iters:
            x = self.val_iters[-1]
            y = self.val_losses[-1]
            self.latest_val_text = self.ax.annotate(
                f"{y:.4f}",
                xy=(x, y),
                xytext=(6, 0),
                textcoords="offset points",
                color="tab:orange",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )

    def _autoscale_with_padding(self):
        # Compute limits from currently visible data
        xs = list(self.iterations)
        ys = list(self.train_losses_display)

        if len(self.val_iters) > 0:
            xs += self.val_iters
            ys += self.val_losses

        if not xs or not ys:
            return

        # Filter non-positive for log-scale
        if self.logy:
            pos_ys = [v for v in ys if v > 0]
            if len(pos_ys) == 0:
                self.ax.set_yscale("linear") # Fallback
                y_min, y_max = min(ys), max(ys)
            else:
                self.ax.set_yscale("log") # Ensure it's set
                y_min, y_max = min(pos_ys), max(pos_ys)
        else:
            y_min, y_max = min(ys), max(ys)

        x_min, x_max = min(xs), max(xs)

        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        if y_min == y_max:
            eps = 1e-6 if self.logy else max(1e-6, abs(y_min) * 0.05)
            y_min -= eps
            y_max += eps

        # Apply padding
        y_range = y_max - y_min
        y_pad = self.ylim_padding * y_range
        
        # --- CHANGE 5: Add padding to x-axis ---
        x_range = x_max - x_min
        x_pad = self.ylim_padding * x_range

        # Set limits with padding
        # Use max(0, ...) for x_min to avoid negative iterations on plot
        self.ax.set_xlim(max(0, x_min - x_pad), x_max + x_pad)
        self.ax.set_ylim(y_min - y_pad, y_max + y_pad)