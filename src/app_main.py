"""Tkinter UI: multi-image document correction, per-image corners, batch export."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from tkinter import filedialog, messagebox, ttk
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from .detect_document import default_corners_image
from .onnx_backend import detect_quad_fused
from .rectify import optional_clahe_bgr, save_image, warp_document

# Padding inside canvas when fitting image (pixels per side)
_CANVAS_IMAGE_PAD = 6

HANDLE_RADIUS_DISP = 9


def resource_path(*parts: str) -> str:
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base, *parts)


def batch_filename_for_index(yyyymmdd: str, index_one_based: int, ext: str) -> str:
    """IMG_YYYYMMDD_NNN.ext e.g. IMG_20260406_001.png"""
    return f"IMG_{yyyymmdd}_{index_one_based:03d}.{ext.lstrip('.')}"


@dataclass
class LoadedImage:
    path: str
    image_bgr: np.ndarray
    corners: List[Tuple[float, float]] = field(default_factory=list)
    warped: Optional[np.ndarray] = None
    detection_done: bool = False
    corners_user_edited: bool = False


class DocAdjApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PicAdj — 文档透视矫正")
        self.root.minsize(960, 620)

        self._items: List[LoadedImage] = []
        self._current: int = -1
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._photo2: Optional[ImageTk.PhotoImage] = None
        self._disp_scale = 1.0
        self._disp_off = (0.0, 0.0)
        self._drag_idx: Optional[int] = None
        self._drag_did_move = False
        self._src_bg_id: Optional[int] = None
        self._src_line_ids: List[int] = []
        self._src_handle_ids: List[int] = []
        self._listbox: Optional[tk.Listbox] = None
        self._redraw_after_id: Optional[Any] = None
        self._detection_queue: deque[int] = deque()
        self._detection_after_id: Optional[Any] = None

        self._var_clahe = tk.BooleanVar(value=False)
        self._var_try_onnx = tk.BooleanVar(value=True)
        self._var_batch_fmt = tk.StringVar(value="jpg")  # png | jpg

        self._build_ui()

    def _current_item(self) -> Optional[LoadedImage]:
        if 0 <= self._current < len(self._items):
            return self._items[self._current]
        return None

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="导入图片…", command=self._on_import).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="移除所选", command=self._on_remove_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="清空列表", command=self._on_clear_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="重新检测", command=self._on_redetect).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="应用矫正（当前）", command=self._on_warp_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="全部应用并保存…", command=self._on_batch_warp_save).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(top, text="保存当前为…", command=self._on_save_current_as).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(top, text="CLAHE 增强", variable=self._var_clahe).pack(
            side=tk.LEFT, padx=(12, 2)
        )
        ttk.Checkbutton(
            top, text="低置信度时尝试 ONNX（若已放置模型）", variable=self._var_try_onnx
        ).pack(side=tk.LEFT, padx=(8, 2))

        ttk.Label(top, text="批量格式:").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Radiobutton(top, text="JPEG", variable=self._var_batch_fmt, value="jpg").pack(
            side=tk.LEFT
        )
        ttk.Radiobutton(top, text="PNG", variable=self._var_batch_fmt, value="png").pack(
            side=tk.LEFT
        )

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        list_frame = ttk.Frame(body, width=220)
        list_frame.pack_propagate(False)
        body.add(list_frame, weight=0)

        ttk.Label(
            list_frame,
            text="已导入（单击切换；Ctrl/Shift 多选；「移除所选」可删多张）",
        ).pack(anchor=tk.W, padx=4, pady=(0, 2))
        lb_frame = ttk.Frame(list_frame)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        scroll = ttk.Scrollbar(lb_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox = tk.Listbox(
            lb_frame, height=18, selectmode=tk.EXTENDED, yscrollcommand=scroll.set
        )
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self._listbox.bind("<Delete>", self._on_remove_selected_event)

        mid = ttk.Panedwindow(body, orient=tk.HORIZONTAL)
        body.add(mid, weight=1)

        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=1)
        mid.add(right, weight=1)

        ttk.Label(left, text="原图与角点（拖动圆点微调，每张图独立记忆）").pack(anchor=tk.W)
        self.canvas_src = tk.Canvas(left, bg="#222", highlightthickness=0)
        self.canvas_src.pack(fill=tk.BOTH, expand=True)
        self.canvas_src.bind("<ButtonPress-1>", self._on_press)
        self.canvas_src.bind("<B1-Motion>", self._on_motion)
        self.canvas_src.bind("<ButtonRelease-1>", self._on_release)

        ttk.Label(right, text="矫正预览（选中列表项即自动根据当前角点生成）").pack(anchor=tk.W)
        self.canvas_dst = tk.Canvas(right, bg="#222", highlightthickness=0)
        self.canvas_dst.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="请导入一张或多张图片")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        help_txt = (
            "提示：导入后先尽快加入列表，自动检测在后台排队执行；可拖动角点，已手动调整的图将跳过自动检测。"
            "列表可多选；点「移除所选」或 Delete 可去掉选中项（未选中列表时默认移除当前预览项，均不删磁盘文件）。"
            "「全部应用并保存」按 IMG_YYYYMMDD_三位编号 写入所选文件夹。"
        )
        ttk.Label(self.root, text=help_txt, font=("Segoe UI", 8), foreground="#555").pack(
            side=tk.BOTTOM, anchor=tk.W, padx=8, pady=(0, 4)
        )

        self.root.bind("<Configure>", self._schedule_redraw_fit)
        self.canvas_src.bind("<Configure>", self._schedule_redraw_fit)
        self.canvas_dst.bind("<Configure>", self._schedule_redraw_fit)

    def _schedule_redraw_fit(self, _event: object = None) -> None:
        """Debounce: window/fullscreen/pane resize refits previews to canvas size."""
        if self._redraw_after_id is not None:
            self.root.after_cancel(self._redraw_after_id)
        self._redraw_after_id = self.root.after(80, self._redraw_fit_idle)

    def _redraw_fit_idle(self) -> None:
        self._redraw_after_id = None
        self._redraw_all()

    def _set_status(self, msg: str, detail: str = "") -> None:
        self.status.set(msg if not detail else f"{msg}  |  {detail}")

    def _refresh_listbox(self) -> None:
        if self._listbox is None:
            return
        self._listbox.delete(0, tk.END)
        for it in self._items:
            name = os.path.basename(it.path)
            self._listbox.insert(tk.END, name)
        if self._items and 0 <= self._current < len(self._items):
            self._listbox.selection_clear(0, tk.END)
            self._listbox.selection_set(self._current)
            self._listbox.see(self._current)

    def _set_current(self, index: int) -> None:
        if not self._items:
            self._current = -1
            return
        self._current = max(0, min(index, len(self._items) - 1))
        self._refresh_listbox()
        self._ensure_warp_preview_for_current()
        self._redraw_all()

    def _on_list_select(self, _ev: object = None) -> None:
        if self._listbox is None or not self._items:
            return
        sel_raw = self._listbox.curselection()
        if not sel_raw:
            return
        sel = sorted({int(i) for i in sel_raw if 0 <= int(i) < len(self._items)})
        if not sel:
            return
        if self._current in sel:
            idx = self._current
        else:
            idx = sel[0]
        if idx == self._current:
            return
        self._current = idx
        self._ensure_warp_preview_for_current()
        self._redraw_all()
        self._set_status(f"当前: {os.path.basename(self._items[self._current].path)}")

    def _cancel_detection_queue(self) -> None:
        if self._detection_after_id is not None:
            try:
                self.root.after_cancel(self._detection_after_id)
            except (tk.TclError, ValueError):
                pass
            self._detection_after_id = None
        self._detection_queue.clear()

    def _schedule_next_detection(self) -> None:
        if self._detection_after_id is not None or not self._detection_queue:
            return
        self._detection_after_id = self.root.after(35, self._run_one_queued_detection)

    def _run_one_queued_detection(self) -> None:
        self._detection_after_id = None
        if not self._detection_queue:
            self._set_status("文档自动检测已全部完成")
            return

        idx = self._detection_queue.popleft()
        if idx < 0 or idx >= len(self._items):
            self._schedule_next_detection()
            return

        item = self._items[idx]
        if item.detection_done:
            self._schedule_next_detection()
            return

        if item.corners_user_edited:
            item.detection_done = True
            if idx == self._current:
                self._ensure_warp_preview_for_current()
                self._redraw_all()
            self._set_status(
                "已跳过自动检测（已手动调整角点）",
                os.path.basename(item.path),
            )
            if self._detection_queue:
                self._schedule_next_detection()
            else:
                self._set_status("文档自动检测已全部完成")
            return

        pending_after = len(self._detection_queue)
        self._set_status(
            f"自动检测进行中，本张之后还剩 {pending_after} 张…",
            os.path.basename(item.path),
        )
        self._run_detection_for_item(item)
        item.detection_done = True
        if idx == self._current:
            self._ensure_warp_preview_for_current()
            self._redraw_all()

        if self._detection_queue:
            self._schedule_next_detection()
        else:
            self._set_status("文档自动检测已全部完成")

    def _enqueue_detection(self, indices: List[int]) -> None:
        for i in indices:
            self._detection_queue.append(i)
        if indices:
            n = len(self._detection_queue)
            self._set_status(f"已排队自动检测（约 {n} 张）…")
        self._schedule_next_detection()

    def _load_paths(self, paths: List[str]) -> None:
        added = 0
        new_indices: List[int] = []
        for path in paths:
            if not path or not os.path.isfile(path):
                continue
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                messagebox.showwarning("跳过", f"无法读取: {path}")
                continue
            dc = default_corners_image(img)
            corners = [(float(p[0]), float(p[1])) for p in dc.reshape(4, 2)]
            item = LoadedImage(
                path=path,
                image_bgr=img,
                corners=corners,
                warped=None,
                detection_done=False,
                corners_user_edited=False,
            )
            self._items.append(item)
            new_indices.append(len(self._items) - 1)
            added += 1

        if added == 0:
            return

        if self._current < 0:
            self._current = 0
        else:
            self._current = len(self._items) - 1
        self._set_current(self._current)
        self._set_status(f"已导入 {added} 张（共 {len(self._items)} 张），即将自动检测…")
        self.root.after(50, lambda: self._enqueue_detection(new_indices))

    def _on_import(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择图片（可多选）",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All", "*.*"),
            ],
        )
        if paths:
            self._load_paths(list(paths))

    def _on_clear_list(self) -> None:
        if not self._items:
            return
        if not messagebox.askyesno("确认", "清空全部已导入图片？"):
            return
        self._cancel_detection_queue()
        self._items.clear()
        self._current = -1
        self._refresh_listbox()
        self.canvas_src.delete("all")
        self.canvas_dst.delete("all")
        self._invalidate_src_canvas_layers()
        self._set_status("列表已清空")

    def _reindex_detection_queue_after_remove(self, removed_index: int) -> None:
        new_q: deque[int] = deque()
        for j in self._detection_queue:
            if j == removed_index:
                continue
            new_q.append(j - 1 if j > removed_index else j)
        self._detection_queue = new_q

    def _new_current_after_removals(
        self, old_current: int, removed: set[int], new_len: int
    ) -> int:
        if new_len <= 0:
            return -1
        if old_current not in removed:
            return old_current - sum(1 for r in removed if r < old_current)
        return max(
            0,
            min(old_current - sum(1 for r in removed if r <= old_current), new_len - 1),
        )

    def _on_remove_selected(self) -> None:
        if not self._items:
            messagebox.showinfo("提示", "没有可移除的项")
            return
        if self._listbox is None:
            return
        raw = self._listbox.curselection()
        if raw:
            indices = sorted({int(i) for i in raw if 0 <= int(i) < len(self._items)})
        elif 0 <= self._current < len(self._items):
            indices = [self._current]
        else:
            messagebox.showinfo("提示", "请先在列表中选择要移除的图片")
            return
        if not indices:
            return

        removed_set = set(indices)
        names = [os.path.basename(self._items[i].path) for i in sorted(indices)]
        preview = "\n".join(names[:15])
        if len(names) > 15:
            preview += f"\n… 等共 {len(names)} 项"

        n = len(names)
        if not messagebox.askyesno(
            "确认移除",
            f"从列表中移除以下 {n} 项？\n（不会删除磁盘上的原文件）\n\n{preview}",
        ):
            return

        old_current = self._current
        for idx in sorted(removed_set, reverse=True):
            del self._items[idx]
            self._reindex_detection_queue_after_remove(idx)

        new_len = len(self._items)
        if new_len == 0:
            self._current = -1
            self._cancel_detection_queue()
            self.canvas_src.delete("all")
            self.canvas_dst.delete("all")
            self._invalidate_src_canvas_layers()
            self._refresh_listbox()
            self._set_status("列表已空")
            return

        self._current = self._new_current_after_removals(old_current, removed_set, new_len)
        self._refresh_listbox()
        self._ensure_warp_preview_for_current()
        self._redraw_all()
        self._set_status(
            f"已移除 {n} 项",
            f"当前: {os.path.basename(self._items[self._current].path)}",
        )
        self._schedule_next_detection()

    def _on_remove_selected_event(self, _ev: object = None) -> None:
        self._on_remove_selected()
        return "break"

    def _on_redetect(self) -> None:
        item = self._current_item()
        if item is None:
            return
        qi = self._current
        self._detection_queue = deque(i for i in self._detection_queue if i != qi)
        self._run_detection_for_item(item)
        item.detection_done = True
        item.corners_user_edited = False
        self._ensure_warp_preview_for_current()
        self._redraw_all()
        self._set_status("已重新检测当前图")

    def _run_detection_for_item(self, item: LoadedImage) -> None:
        img = item.image_bgr
        onnx_path = None
        if self._var_try_onnx.get():
            cand = resource_path("models", "doc_segment.onnx")
            if os.path.isfile(cand):
                onnx_path = cand

        if self._var_try_onnx.get():
            det, note = detect_quad_fused(img, onnx_path=onnx_path)
        else:
            from .detect_document import detect_quad_opencv

            det, _ = detect_quad_opencv(img)
            note = (
                f"opencv_only:{det.method}:{det.confidence:.2f}"
                if det
                else ""
            )

        if det is not None:
            item.corners = [(float(p[0]), float(p[1])) for p in det.corners.reshape(4, 2)]
            self._set_status(
                f"{os.path.basename(item.path)} 检测成功（{det.confidence:.2f}）",
                note or f"method={det.method}",
            )
        else:
            dc = default_corners_image(img)
            item.corners = [(float(p[0]), float(p[1])) for p in dc.reshape(4, 2)]
            self._set_status(
                f"{os.path.basename(item.path)} 未可靠检测，已用默认角点",
                note or "no_detection",
            )

    def _compute_display_transform(self, cw: int, ch: int) -> None:
        item = self._current_item()
        if item is None:
            return
        ih, iw = item.image_bgr.shape[:2]
        pad = _CANVAS_IMAGE_PAD
        eff_w = max(float(cw) - 2 * pad, 1.0)
        eff_h = max(float(ch) - 2 * pad, 1.0)
        # Fit entire image in the canvas; do not upscale past 1:1 pixels
        scale = min(eff_w / max(iw, 1), eff_h / max(ih, 1), 1.0)
        scale = max(0.05, float(scale))
        dw = float(iw) * scale
        dh = float(ih) * scale
        off_x = (float(cw) - dw) / 2.0
        off_y = (float(ch) - dh) / 2.0
        self._disp_scale = scale
        self._disp_off = (off_x, off_y)

    def _img_to_disp(self, ix: float, iy: float) -> Tuple[float, float]:
        ox, oy = self._disp_off
        return ix * self._disp_scale + ox, iy * self._disp_scale + oy

    def _disp_to_img(self, dx: float, dy: float) -> Tuple[float, float]:
        ox, oy = self._disp_off
        return (dx - ox) / self._disp_scale, (dy - oy) / self._disp_scale

    def _invalidate_src_canvas_layers(self) -> None:
        self._src_bg_id = None
        self._src_line_ids.clear()
        self._src_handle_ids.clear()

    def _redraw_all(self) -> None:
        self._redraw_src()
        self._redraw_dst()

    def _redraw_src(self) -> None:
        self.canvas_src.delete("all")
        self._invalidate_src_canvas_layers()
        item = self._current_item()
        if item is None or len(item.corners) != 4:
            return
        self.canvas_src.update_idletasks()
        cw = max(self.canvas_src.winfo_width(), 320)
        ch = max(self.canvas_src.winfo_height(), 240)
        self._compute_display_transform(cw, ch)
        ih, iw = item.image_bgr.shape[:2]
        rgb = cv2.cvtColor(item.image_bgr, cv2.COLOR_BGR2RGB)
        scale = self._disp_scale
        small = cv2.resize(
            rgb,
            (int(round(iw * scale)), int(round(ih * scale))),
            interpolation=cv2.INTER_AREA,
        )
        pil = Image.fromarray(small)
        self._photo = ImageTk.PhotoImage(pil)
        ox, oy = self._disp_off
        self._src_bg_id = self.canvas_src.create_image(ox, oy, anchor=tk.NW, image=self._photo)

        poly: List[float] = []
        for cx, cy in item.corners:
            dx, dy = self._img_to_disp(cx, cy)
            poly.extend([dx, dy])
        for a in range(4):
            a2 = (a + 1) % 4
            lid = self.canvas_src.create_line(
                poly[a * 2],
                poly[a * 2 + 1],
                poly[a2 * 2],
                poly[a2 * 2 + 1],
                fill="#0cf",
                width=2,
            )
            self._src_line_ids.append(lid)
        for cx, cy in item.corners:
            dx, dy = self._img_to_disp(cx, cy)
            r = HANDLE_RADIUS_DISP
            hid = self.canvas_src.create_oval(
                dx - r, dy - r, dx + r, dy + r, fill="#fc0", outline="#333"
            )
            self._src_handle_ids.append(hid)

    def _redraw_src_handles_only(self) -> None:
        """Move quad overlay only; avoids full bitmap refresh during drag (reduces flicker)."""
        item = self._current_item()
        if item is None or len(item.corners) != 4:
            return
        self.canvas_src.update_idletasks()
        cw = max(self.canvas_src.winfo_width(), 320)
        ch = max(self.canvas_src.winfo_height(), 240)
        self._compute_display_transform(cw, ch)
        if (
            self._src_bg_id is None
            or len(self._src_line_ids) != 4
            or len(self._src_handle_ids) != 4
        ):
            self._redraw_src()
            return
        poly: List[float] = []
        for cx, cy in item.corners:
            dx, dy = self._img_to_disp(cx, cy)
            poly.extend([dx, dy])
        try:
            for a in range(4):
                a2 = (a + 1) % 4
                self.canvas_src.coords(
                    self._src_line_ids[a],
                    poly[a * 2],
                    poly[a * 2 + 1],
                    poly[a2 * 2],
                    poly[a2 * 2 + 1],
                )
            for i, (cx, cy) in enumerate(item.corners):
                dx, dy = self._img_to_disp(cx, cy)
                r = HANDLE_RADIUS_DISP
                self.canvas_src.coords(
                    self._src_handle_ids[i], dx - r, dy - r, dx + r, dy + r
                )
        except tk.TclError:
            self._redraw_src()

    def _redraw_dst(self) -> None:
        self.canvas_dst.delete("all")
        item = self._current_item()
        if item is None or item.warped is None:
            return
        self.canvas_dst.update_idletasks()
        cw = max(self.canvas_dst.winfo_width(), 320)
        ch = max(self.canvas_dst.winfo_height(), 240)
        rgb = cv2.cvtColor(item.warped, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        pad = _CANVAS_IMAGE_PAD
        eff_w = max(float(cw) - 2 * pad, 1.0)
        eff_h = max(float(ch) - 2 * pad, 1.0)
        sc = min(eff_w / max(w0, 1), eff_h / max(h0, 1), 1.0)
        sc = max(0.05, float(sc))
        w1, h1 = int(round(w0 * sc)), int(round(h0 * sc))
        sm = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_AREA)
        self._photo2 = ImageTk.PhotoImage(Image.fromarray(sm))
        self.canvas_dst.create_image(cw // 2, ch // 2, image=self._photo2)

    def _hit_test(self, dx: float, dy: float) -> Optional[int]:
        item = self._current_item()
        if item is None or len(item.corners) != 4:
            return None
        for i, (cx, cy) in enumerate(item.corners):
            px, py = self._img_to_disp(cx, cy)
            if (px - dx) ** 2 + (py - dy) ** 2 <= (HANDLE_RADIUS_DISP + 6) ** 2:
                return i
        return None

    def _on_press(self, ev: tk.Event) -> None:
        self._drag_idx = self._hit_test(ev.x, ev.y)
        self._drag_did_move = False

    def _on_motion(self, ev: tk.Event) -> None:
        item = self._current_item()
        if self._drag_idx is None or item is None or len(item.corners) != 4:
            return
        self.canvas_src.update_idletasks()
        cw = max(self.canvas_src.winfo_width(), 320)
        ch = max(self.canvas_src.winfo_height(), 240)
        self._compute_display_transform(cw, ch)
        ix, iy = self._disp_to_img(ev.x, ev.y)
        ih, iw = item.image_bgr.shape[:2]
        ix = float(max(0, min(iw - 1, ix)))
        iy = float(max(0, min(ih - 1, iy)))
        if not self._drag_did_move:
            self._drag_did_move = True
            if item.warped is not None:
                item.warped = None
                self._redraw_dst()
        item.corners[self._drag_idx] = (ix, iy)
        item.corners_user_edited = True
        self._redraw_src_handles_only()

    def _on_release(self, ev: tk.Event) -> None:  # noqa: ARG002
        idx = self._drag_idx
        moved = self._drag_did_move
        self._drag_idx = None
        self._drag_did_move = False
        if idx is None or not moved:
            return
        item = self._current_item()
        if item is None or len(item.corners) != 4:
            return
        item.warped = self._warp_item(item)
        self._redraw_dst()
        self._set_status("角点已更新；矫正与右侧预览在松开鼠标后计算")

    def _warp_item(self, item: LoadedImage) -> np.ndarray:
        pts = np.array(item.corners, dtype=np.float32)
        out = warp_document(item.image_bgr, pts)
        if self._var_clahe.get():
            out = optional_clahe_bgr(out)
        return out

    def _ensure_warp_preview_for_current(self) -> None:
        """Compute rectified preview for the active list item (right pane)."""
        item = self._current_item()
        if item is None or len(item.corners) != 4:
            return
        item.warped = self._warp_item(item)

    def _on_warp_current(self) -> None:
        item = self._current_item()
        if item is None or len(item.corners) != 4:
            messagebox.showwarning("提示", "请先导入图片并确保当前图有四个角点")
            return
        item.warped = self._warp_item(item)
        self._set_status("当前图矫正完成，可保存或继续编辑其它图")
        self._redraw_dst()

    def _on_batch_warp_save(self) -> None:
        if not self._items:
            messagebox.showwarning("提示", "请先导入图片")
            return
        bad = [i for i, it in enumerate(self._items) if len(it.corners) != 4]
        if bad:
            messagebox.showwarning(
                "提示", f"有 {len(bad)} 张图角点不完整，请先在各图上调整四角后再批量导出"
            )
            return
        pending = [i for i, it in enumerate(self._items) if not it.detection_done]
        if pending:
            sample = [os.path.basename(self._items[i].path) for i in pending[:10]]
            lines = "\n".join(sample)
            extra = (
                f"\n… 等共 {len(pending)} 张"
                if len(pending) > len(sample)
                else ""
            )
            if not messagebox.askyesno(
                "自动检测尚未完成",
                f"以下 {len(pending)} 张尚未完成自动文档检测，"
                f"导出时将使用当前角点（含默认四角或您已拖动的位置）：\n\n"
                f"{lines}{extra}\n\n是否继续批量导出？",
            ):
                return
        folder = filedialog.askdirectory(title="选择保存文件夹")
        if not folder:
            return
        ext = self._var_batch_fmt.get().strip().lower()
        if ext not in ("png", "jpg", "jpeg"):
            ext = "png"
        if ext == "jpeg":
            ext = "jpg"
        prefix = date.today().strftime("%Y%m%d")
        n_ok = 0
        for i, item in enumerate(self._items, start=1):
            warped = self._warp_item(item)
            fname = batch_filename_for_index(prefix, i, ext)
            out_path = os.path.join(folder, fname)
            try:
                save_image(out_path, warped)
                item.warped = warped
                n_ok += 1
            except OSError as e:
                messagebox.showerror("错误", f"写入失败 {out_path}\n{e}")
                return
        self._redraw_dst()
        self._set_status(
            f"已批量保存 {n_ok} 张至 {folder}",
            f"示例 IMG_{prefix}_001…",
        )

    def _on_save_current_as(self) -> None:
        item = self._current_item()
        if item is None:
            return
        if item.warped is None:
            if len(item.corners) != 4:
                messagebox.showwarning("提示", "当前图缺少四个角点")
                return
            item.warped = self._warp_item(item)
            self._redraw_dst()
        prefix = date.today().strftime("%Y%m%d")
        idx = self._current + 1
        ext = self._var_batch_fmt.get().strip().lower()
        if ext == "jpeg":
            ext = "jpg"
        if ext not in ("png", "jpg"):
            ext = "png"
        initial = batch_filename_for_index(prefix, idx, ext)
        path = filedialog.asksaveasfilename(
            title="保存当前",
            initialfile=initial,
            defaultextension=f".{ext}",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        try:
            save_image(path, item.warped)
            self._set_status(f"已保存: {path}")
        except OSError as e:
            messagebox.showerror("错误", str(e))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = DocAdjApp()
    app.run()


if __name__ == "__main__":
    main()
