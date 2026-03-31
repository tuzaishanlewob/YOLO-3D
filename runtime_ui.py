import time

import cv2

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


VERSION_ORDER = ("v1", "v2", "v3", "v4")
VERSION_CHECKBOX_LABELS = {
    "v1": "V1 - GT reference",
    "v2": "V2 - depth map",
    "v3": "V3 - YOLO + model depth",
    "v4": "V4 - segmentation",
}


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def short_airsim_class_name(object_name):
    """Convert full AirSim object name to a compact class label."""
    if object_name is None:
        return "object"
    name = str(object_name).strip()
    if not name:
        return "object"
    return name.split("_", 1)[0]


def get_version_selection(config):
    """Read explicit version toggles, falling back to legacy compare flags."""
    return {
        "v1": to_bool(config.get("enable_v1", config.get("use_airsim_ground_truth", True))),
        "v2": to_bool(config.get("enable_v2", True)),
        "v3": to_bool(config.get("enable_v3", config.get("compare_three_versions", False))),
        "v4": to_bool(config.get("enable_v4", config.get("compare_four_versions", False))),
    }


def apply_version_selection(config):
    """Normalize runtime config so explicit V1-V4 toggles and legacy flags stay aligned."""
    normalized = dict(config)
    versions = get_version_selection(normalized)
    for version_key, enabled in versions.items():
        normalized[f"enable_{version_key}"] = bool(enabled)
    normalized["compare_three_versions"] = bool(versions["v3"])
    normalized["compare_four_versions"] = bool(versions["v4"])
    if versions["v4"]:
        normalized["enable_segmentation"] = True
    return normalized


def launch_config_ui(defaults):
    """Show a small startup UI for configuring run options."""
    if tk is None:
        print("Tkinter is unavailable; using built-in defaults")
        return defaults

    root = tk.Tk()
    root.title("YOLO-3D Run Config")
    root.geometry("820x700")
    root.resizable(False, False)

    container = ttk.Frame(root, padding=8)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    form_frame = ttk.Frame(canvas)

    form_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=form_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    container.columnconfigure(0, weight=1)
    container.rowconfigure(0, weight=1)

    fields = [
        ("integrated_preview_ui", "bool"),
        ("show_cv2_windows", "bool"),
        ("use_airsim_source", "bool"),
        ("use_airsim_ground_truth", "bool"),
        ("gt_detection_mesh_pattern", "str"),
        ("gt_detection_radius_m", "float"),
        ("gt_use_depthplanar", "bool"),
        ("versions", "versions"),
        ("source", "str"),
        ("output_path", "str"),
        ("airsim_camera_name", "str"),
        ("airsim_vehicle_name", "str"),
        ("use_airsim_camera_info", "bool"),
        ("airsim_refresh_camera_params_every_frame", "bool"),
        ("camera_params_file", "str"),
        ("yolo_model_size", "str"),
        ("yolo_weights", "str"),
        ("depth_model_size", "str"),
        ("device", "int"),
        ("conf_threshold", "float"),
        ("iou_threshold", "float"),
        ("enable_tracking", "bool"),
        ("enable_bev", "bool"),
        ("enable_pseudo_3d", "bool"),
        ("enable_stream", "bool"),
        ("enable_segmentation", "bool"),
        ("export_dataset", "bool"),
        ("dataset_root", "str"),
        ("hdf5_include_segmentation", "bool"),
    ]

    vars_map = {}
    row = 0
    version_defaults = get_version_selection(defaults)
    for key, kind in fields:
        label = ttk.Label(form_frame, text=key)
        label.grid(row=row, column=0, padx=6, pady=4, sticky="w")

        if kind == "versions":
            widget = ttk.LabelFrame(form_frame, text="Version Selection", padding=(8, 6))
            widget.grid(row=row, column=1, padx=6, pady=4, sticky="ew")
            for idx, version_key in enumerate(VERSION_ORDER):
                v = tk.BooleanVar(value=version_defaults[version_key])
                version_box = ttk.Checkbutton(
                    widget,
                    text=VERSION_CHECKBOX_LABELS[version_key],
                    variable=v,
                )
                version_box.grid(row=idx // 2, column=idx % 2, padx=4, pady=3, sticky="w")
                vars_map[f"enable_{version_key}"] = (v, "bool")
            ttk.Label(
                widget,
                text="V1 is the AirSim ground-truth reference. Selecting V4 will turn segmentation on automatically.",
                justify="left",
                wraplength=420,
            ).grid(row=2, column=0, columnspan=2, padx=4, pady=(6, 0), sticky="w")
            widget.columnconfigure(0, weight=1)
            widget.columnconfigure(1, weight=1)
            row += 1
            continue

        default_value = defaults.get(key)
        if kind == "bool":
            v = tk.BooleanVar(value=to_bool(default_value))
            widget = ttk.Checkbutton(form_frame, variable=v)
        else:
            v = tk.StringVar(value=str(default_value))
            widget = ttk.Entry(form_frame, textvariable=v, width=38)
        widget.grid(row=row, column=1, padx=6, pady=4, sticky="ew")
        vars_map[key] = (v, kind)
        row += 1

    form_frame.columnconfigure(1, weight=1)

    result = {"accepted": False}

    def on_start():
        parsed = {}
        for key, (var, kind) in vars_map.items():
            raw = var.get()
            if kind == "bool":
                parsed[key] = to_bool(raw)
            elif kind == "int":
                try:
                    parsed[key] = int(raw)
                except ValueError:
                    parsed[key] = int(defaults[key])
            elif kind == "float":
                try:
                    parsed[key] = float(raw)
                except ValueError:
                    parsed[key] = float(defaults[key])
            else:
                parsed[key] = str(raw)

        if parsed["source"].isdigit():
            parsed["source"] = int(parsed["source"])

        parsed = apply_version_selection(parsed)
        result["accepted"] = True
        result["config"] = parsed
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frame = ttk.Frame(root, padding=(8, 0, 8, 8))
    btn_frame.pack(fill="x")
    ttk.Button(btn_frame, text="Start", command=on_start).pack(side="left", padx=4)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=4)

    root.mainloop()

    if result.get("accepted"):
        return result["config"]

    print("Config UI canceled; using built-in defaults")
    return defaults


class RuntimeDashboard:
    """Runtime dashboard with control/info panel and live preview panes."""

    def __init__(self, enabled=True, selected_versions=None, preview_interval_s=0.12, table_interval_s=0.25):
        self.enabled = bool(enabled and tk is not None and ttk is not None and Image is not None and ImageTk is not None)
        self.stop_requested = False
        visible_versions = tuple(v for v in VERSION_ORDER if selected_versions is None or v in selected_versions)
        self.visible_versions = visible_versions
        self.preview_interval_s = max(0.05, float(preview_interval_s))
        self.table_interval_s = max(self.preview_interval_s, float(table_interval_s))
        self._last_preview_update = 0.0
        self._last_table_update = 0.0
        self._last_status_text = None
        self._last_metrics_text = None
        self._last_object_rows = ()

        if not self.enabled:
            self.root = None
            return

        self.root = tk.Tk()
        self.root.title("YOLO-3D Dashboard")
        self.root.geometry("1360x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        top = ttk.Frame(self.root, padding=8)
        top.pack(side="top", fill="both", expand=True)
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side="bottom", fill="x")

        ttk.Label(top, text="Runtime Status", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.status_var = tk.StringVar(value="Starting...")
        self.metrics_var = tk.StringVar(value="Metrics: --")
        ttk.Label(top, textvariable=self.status_var, justify="left", wraplength=1200).pack(anchor="w", pady=4)
        ttk.Label(top, textvariable=self.metrics_var, justify="left", wraplength=1200).pack(anchor="w", pady=4)
        ttk.Button(top, text="Stop", command=self._on_close).pack(anchor="w", pady=(12, 8))

        ttk.Label(top, text="Per-object Metrics", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 4))
        table_frame = ttk.Frame(top)
        table_frame.pack(fill="both", expand=True)

        self.table_columns = (
            "obj",
            "gt_world",
            "v2_depth", "v2_world", "v2_uv",
            "v3_depth", "v3_world", "v3_uv",
            "v4_depth", "v4_world", "v4_uv",
        )
        self.metrics_table = ttk.Treeview(table_frame, columns=self.table_columns, show="headings", height=14)
        display_columns = ["obj"]
        if "v1" in self.visible_versions:
            display_columns.append("gt_world")
        if "v2" in self.visible_versions:
            display_columns.extend(("v2_depth", "v2_world", "v2_uv"))
        if "v3" in self.visible_versions:
            display_columns.extend(("v3_depth", "v3_world", "v3_uv"))
        if "v4" in self.visible_versions:
            display_columns.extend(("v4_depth", "v4_world", "v4_uv"))
        self.metrics_table.configure(displaycolumns=display_columns)
        self.metrics_table.heading("obj", text="Object")
        self.metrics_table.heading("gt_world", text="V1 GT World(x,y,z)")
        self.metrics_table.heading("v2_depth", text="V2 Depth(m)")
        self.metrics_table.heading("v2_world", text="V2 World(x,y,z)")
        self.metrics_table.heading("v2_uv", text="V2 2D(u,v)")
        self.metrics_table.heading("v3_depth", text="V3 Depth(m)")
        self.metrics_table.heading("v3_world", text="V3 World(x,y,z)")
        self.metrics_table.heading("v3_uv", text="V3 2D(u,v)")
        self.metrics_table.heading("v4_depth", text="V4 Depth(m)")
        self.metrics_table.heading("v4_world", text="V4 World(x,y,z)")
        self.metrics_table.heading("v4_uv", text="V4 2D(u,v)")

        self.metrics_table.column("obj", width=130, anchor="w")
        self.metrics_table.column("gt_world", width=220, anchor="w")
        self.metrics_table.column("v2_depth", width=90, anchor="center")
        self.metrics_table.column("v2_world", width=220, anchor="w")
        self.metrics_table.column("v2_uv", width=115, anchor="w")
        self.metrics_table.column("v3_depth", width=90, anchor="center")
        self.metrics_table.column("v3_world", width=220, anchor="w")
        self.metrics_table.column("v3_uv", width=115, anchor="w")
        self.metrics_table.column("v4_depth", width=90, anchor="center")
        self.metrics_table.column("v4_world", width=220, anchor="w")
        self.metrics_table.column("v4_uv", width=115, anchor="w")

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.metrics_table.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.metrics_table.xview)
        self.metrics_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.metrics_table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.metrics_table.bind("<MouseWheel>", self._on_table_mousewheel, add="+")
        self.metrics_table.bind("<Shift-MouseWheel>", self._on_table_shift_mousewheel, add="+")

        self.image_labels = {}
        pane_titles = [
            ("result", "3D Object Detection"),
            ("detection", "Object Detection"),
            ("depth", "Depth Map"),
        ]
        if "v4" in self.visible_versions:
            pane_titles.append(("v4", "V4 Segmentation"))

        for idx, (key, title) in enumerate(pane_titles):
            pane = ttk.Frame(bottom, padding=4)
            pane.grid(row=0, column=idx, sticky="nsew")
            ttk.Label(pane, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            lbl = ttk.Label(pane)
            lbl.pack(fill="both", expand=True)
            self.image_labels[key] = lbl

        for idx in range(len(pane_titles)):
            bottom.columnconfigure(idx, weight=1)

    def _on_close(self):
        self.stop_requested = True

    def _wheel_units(self, event):
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return 0
        if abs(delta) >= 120:
            return -int(delta / 120)
        return -1 if delta > 0 else 1

    def _on_table_mousewheel(self, event):
        units = self._wheel_units(event)
        if units:
            if getattr(event, "state", 0) & 0x0001:
                self.metrics_table.xview_scroll(units, "units")
            else:
                self.metrics_table.yview_scroll(units, "units")
        return "break"

    def _on_table_shift_mousewheel(self, event):
        units = self._wheel_units(event)
        if units:
            self.metrics_table.xview_scroll(units, "units")
        return "break"

    def process_events(self):
        if not self.enabled or self.root is None:
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
            return not self.stop_requested
        except tk.TclError:
            self.stop_requested = True
            self.enabled = False
            self.root = None
            return False

    def _set_image(self, key, frame_bgr):
        if not self.enabled:
            return

        label = self.image_labels.get(key)
        if label is None:
            return

        if frame_bgr is None:
            label.configure(image="", text="No preview", compound="center")
            label.image = None
            return

        h, w = frame_bgr.shape[:2]
        # Keep bottom previews compact so they don't dominate the dashboard.
        max_w = 320 if len(self.image_labels) <= 3 else 250
        max_h = 220
        scale = min(max_w / max(1, w), max_h / max(1, h))
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))
        preview = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=photo, text="")
        label.image = photo

    def _refresh_table(self, object_rows):
        if not hasattr(self, "metrics_table"):
            return

        rows_tuple = tuple(object_rows or ())
        xview = self.metrics_table.xview()
        yview = self.metrics_table.yview()
        for item in self.metrics_table.get_children():
            self.metrics_table.delete(item)
        for row in rows_tuple:
            self.metrics_table.insert("", "end", values=row)
        if xview:
            self.metrics_table.xview_moveto(xview[0])
        if yview:
            self.metrics_table.yview_moveto(yview[0])
        self._last_object_rows = rows_tuple

    def update(
        self,
        result_frame=None,
        detection_frame=None,
        depth_frame=None,
        v4_frame=None,
        status_text="",
        metrics_text="",
        object_rows=None,
    ):
        if not self.enabled or self.root is None:
            return

        now = time.monotonic()
        if status_text != self._last_status_text:
            self.status_var.set(status_text or "--")
            self._last_status_text = status_text
        if metrics_text != self._last_metrics_text:
            self.metrics_var.set(metrics_text or "Metrics: --")
            self._last_metrics_text = metrics_text

        self.process_events()
        if self.stop_requested:
            return

        if now - self._last_preview_update >= self.preview_interval_s:
            self._set_image("result", result_frame)
            self._set_image("detection", detection_frame)
            self._set_image("depth", depth_frame)
            self._set_image("v4", v4_frame)
            self._last_preview_update = now

        rows_tuple = tuple(object_rows or ())
        table_due = (now - self._last_table_update) >= self.table_interval_s
        rows_changed = rows_tuple != self._last_object_rows
        if (rows_changed and (table_due or not self._last_object_rows)) or table_due:
            self._refresh_table(rows_tuple)
            self._last_table_update = now

        self.process_events()

    def close(self):
        if self.enabled and self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
