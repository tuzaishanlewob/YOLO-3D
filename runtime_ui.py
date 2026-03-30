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


def launch_config_ui(defaults):
    """Show a small startup UI for configuring run options."""
    if tk is None:
        print("Tkinter is unavailable; using built-in defaults")
        return defaults

    root = tk.Tk()
    root.title("YOLO-3D Run Config")
    root.resizable(False, False)

    fields = [
        ("integrated_preview_ui", "bool"),
        ("show_cv2_windows", "bool"),
        ("use_airsim_source", "bool"),
        ("use_airsim_ground_truth", "bool"),
        ("gt_detection_mesh_pattern", "str"),
        ("gt_detection_radius_m", "float"),
        ("gt_use_depthplanar", "bool"),
        ("compare_three_versions", "bool"),
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
    ]

    vars_map = {}
    row = 0
    for key, kind in fields:
        label = ttk.Label(root, text=key)
        label.grid(row=row, column=0, padx=6, pady=4, sticky="w")
        default_value = defaults.get(key)
        if kind == "bool":
            v = tk.BooleanVar(value=to_bool(default_value))
            widget = ttk.Checkbutton(root, variable=v)
        else:
            v = tk.StringVar(value=str(default_value))
            widget = ttk.Entry(root, textvariable=v, width=48)
        widget.grid(row=row, column=1, padx=6, pady=4, sticky="ew")
        vars_map[key] = (v, kind)
        row += 1

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

        result["accepted"] = True
        result["config"] = parsed
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=row, column=0, columnspan=2, pady=8)
    ttk.Button(btn_frame, text="Start", command=on_start).pack(side="left", padx=4)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=4)

    root.mainloop()

    if result.get("accepted"):
        return result["config"]

    print("Config UI canceled; using built-in defaults")
    return defaults


class RuntimeDashboard:
    """Runtime dashboard with control/info panel and live preview panes."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled and tk is not None and ttk is not None and Image is not None and ImageTk is not None)
        self.stop_requested = False

        if not self.enabled:
            self.root = None
            return

        self.root = tk.Tk()
        self.root.title("YOLO-3D Dashboard")
        self.root.geometry("1480x860")
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
        )
        self.metrics_table = ttk.Treeview(table_frame, columns=self.table_columns, show="headings", height=14)
        self.metrics_table.heading("obj", text="Object")
        self.metrics_table.heading("gt_world", text="V1 GT World(x,y,z)")
        self.metrics_table.heading("v2_depth", text="V2 Depth(m)")
        self.metrics_table.heading("v2_world", text="V2 World(x,y,z)")
        self.metrics_table.heading("v2_uv", text="V2 2D(u,v)")
        self.metrics_table.heading("v3_depth", text="V3 Depth(m)")
        self.metrics_table.heading("v3_world", text="V3 World(x,y,z)")
        self.metrics_table.heading("v3_uv", text="V3 2D(u,v)")

        self.metrics_table.column("obj", width=130, anchor="w")
        self.metrics_table.column("gt_world", width=220, anchor="w")
        self.metrics_table.column("v2_depth", width=90, anchor="center")
        self.metrics_table.column("v2_world", width=220, anchor="w")
        self.metrics_table.column("v2_uv", width=115, anchor="w")
        self.metrics_table.column("v3_depth", width=90, anchor="center")
        self.metrics_table.column("v3_world", width=220, anchor="w")
        self.metrics_table.column("v3_uv", width=115, anchor="w")

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.metrics_table.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.metrics_table.xview)
        self.metrics_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.metrics_table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.image_labels = {}
        pane_titles = [
            ("result", "3D Object Detection"),
            ("detection", "Object Detection"),
            ("depth", "Depth Map"),
        ]

        for idx, (key, title) in enumerate(pane_titles):
            pane = ttk.Frame(bottom, padding=4)
            pane.grid(row=0, column=idx, sticky="nsew")
            ttk.Label(pane, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            lbl = ttk.Label(pane)
            lbl.pack(fill="both", expand=True)
            self.image_labels[key] = lbl

        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.columnconfigure(2, weight=1)

    def _on_close(self):
        self.stop_requested = True

    def _set_image(self, key, frame_bgr):
        if not self.enabled or frame_bgr is None:
            return

        label = self.image_labels.get(key)
        if label is None:
            return

        h, w = frame_bgr.shape[:2]
        # Keep bottom previews compact so they don't dominate the dashboard.
        max_w = 360
        max_h = 220
        scale = min(max_w / max(1, w), max_h / max(1, h))
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))
        preview = cv2.resize(frame_bgr, (target_w, target_h))
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=photo)
        label.image = photo

    def update(self, result_frame=None, detection_frame=None, depth_frame=None, status_text="", metrics_text="", object_rows=None):
        if not self.enabled or self.root is None:
            return

        self.status_var.set(status_text or "--")
        self.metrics_var.set(metrics_text or "Metrics: --")

        self._set_image("result", result_frame)
        self._set_image("detection", detection_frame)
        self._set_image("depth", depth_frame)

        if hasattr(self, "metrics_table"):
            for item in self.metrics_table.get_children():
                self.metrics_table.delete(item)
            for row in (object_rows or []):
                self.metrics_table.insert("", "end", values=row)

        self.root.update_idletasks()
        self.root.update()

    def close(self):
        if self.enabled and self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
