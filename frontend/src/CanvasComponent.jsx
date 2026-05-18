import { useEffect, useRef, useState } from "react";
import * as fabric from "fabric";

const SIDEBAR_W = 224; // w-56 = 14rem = 224px
const HEADER_H = 52;   // header height
const TOOLBAR_H = 44;  // toolbar height
const PADDING = 32;    // p-4 * 2

function b64toDataURL(b64) {
  if (b64.startsWith("data:")) return b64;
  return `data:image/png;base64,${b64}`;
}

export default function CanvasComponent({ data }) {
  const canvasRef = useRef(null);
  const fabricRef = useRef(null);
  const [selectedName, setSelectedName] = useState(null);
  const [ready, setReady] = useState(false);

  const {
    background_base64,
    original_width: origW,
    original_height: origH,
    layers,
  } = data;

  // Kích thước vùng chứa canvas thực tế
  const availW = window.innerWidth - SIDEBAR_W - PADDING;
  const availH = window.innerHeight - HEADER_H - TOOLBAR_H - PADDING;

  // 1 scale duy nhất giữ đúng tỉ lệ ảnh gốc
  const scale = Math.min(1, availW / origW, availH / origH);
  const canvasW = Math.floor(origW * scale);
  const canvasH = Math.floor(origH * scale);
  const scaleX = canvasW / origW;  // scale chính xác để background fill đúng canvas
  const scaleY = canvasH / origH;

  useEffect(() => {
    if (!canvasRef.current) return;

    let disposed = false;

    const canvas = new fabric.Canvas(canvasRef.current, {
      width: canvasW,
      height: canvasH,
      selection: true,
      backgroundColor: "#1a1a2e",
    });
    fabricRef.current = canvas;

    canvas.on("selection:created", (e) =>
      setSelectedName(e.selected?.[0]?.layerName ?? null)
    );
    canvas.on("selection:updated", (e) =>
      setSelectedName(e.selected?.[0]?.layerName ?? null)
    );
    canvas.on("selection:cleared", () => setSelectedName(null));
    canvas.on("object:modified", (e) => {
      if (e.target?.layerName === "background") {
        canvas.sendObjectToBack(e.target);
      }
    });

    fabric.FabricImage.fromURL(b64toDataURL(background_base64), {
      crossOrigin: "anonymous",
    })
      .then((bgImg) => {
        if (disposed) return;

        bgImg.set({
          left: 0,
          top: 0,
          originX: "left",
          originY: "top",
          scaleX: scale,  // scale duy nhất, không tính riêng X/Y
          scaleY: scale,
          selectable: true,
          evented: true,
          hasControls: true,
          hasBorders: true,
          borderColor: "#94a3b8",
          cornerColor: "#94a3b8",
          cornerSize: 8,
          transparentCorners: false,
        });
        bgImg.layerName = "background";
        canvas.add(bgImg);
        console.log("=== FABRIC STATE ===");
        console.log("Canvas size:", canvas.width, canvas.height);
        console.log("bgImg left/top:", bgImg.left, bgImg.top);
        console.log("bgImg scaleX/Y:", bgImg.scaleX, bgImg.scaleY);
        console.log("bgImg width/height sau scale:", bgImg.getScaledWidth(), bgImg.getScaledHeight());
        console.log("bgImg getBoundingRect:", bgImg.getBoundingRect());
        canvas.sendObjectToBack(bgImg);

        const loadLayer = (index) => {
          if (disposed) return;
          if (index >= layers.length) {
            canvas.renderAll();
            setReady(true);
            return;
          }

          const layer = layers[index];

          fabric.FabricImage.fromURL(b64toDataURL(layer.png_base64), {
            crossOrigin: "anonymous",
          })
            .then((img) => {
              if (disposed) return;
              img.set({
                left: layer.x * scale,
                top: layer.y * scale,
                originX: "left",
                originY: "top",
                scaleX: scaleX,
                scaleY: scaleY,
                selectable: true,
                hasControls: true,
                hasBorders: true,
                borderColor: "#6366f1",
                cornerColor: "#6366f1",
                cornerSize: 8,
                transparentCorners: false,
              });
              img.layerName = layer.keyword;
              canvas.add(img);
              canvas.bringObjectToFront(img);
              loadLayer(index + 1);
            })
            .catch(() => loadLayer(index + 1));
        };

        loadLayer(0);
      })
      .catch((err) => console.error("Lỗi load background:", err));

    return () => {
      disposed = true;
      canvas.dispose();
      fabricRef.current = null;
    };
  }, [canvasW, canvasH, scale, data]);

  const bringForward = () => {
    const o = fabricRef.current?.getActiveObject();
    if (o) { fabricRef.current.bringObjectForward(o); fabricRef.current.renderAll(); }
  };
  const sendBackward = () => {
    const o = fabricRef.current?.getActiveObject();
    if (o && o.layerName !== "background") {
      fabricRef.current.sendObjectBackwards(o);
      const bg = fabricRef.current.getObjects().find((obj) => obj.layerName === "background");
      if (bg) fabricRef.current.sendObjectToBack(bg);
      fabricRef.current.renderAll();
    }
  };
  const bringToFront = () => {
    const o = fabricRef.current?.getActiveObject();
    if (o && o.layerName !== "background") {
      fabricRef.current.bringObjectToFront(o);
      fabricRef.current.renderAll();
    }
  };
  const exportCanvas = () => {
    const dataURL = fabricRef.current?.toDataURL({
      format: "png",
      multiplier: 1 / scale,
    });
    if (!dataURL) return;
    const a = document.createElement("a");
    a.href = dataURL;
    a.download = "magic-canvas-export.png";
    a.click();
  };

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Toolbar */}
      <div className="flex items-center gap-2 bg-gray-900 rounded-xl px-3 py-2 border border-gray-800">
        <span className="text-gray-400 text-xs mr-1">
          {selectedName ? `✦ ${selectedName}` : "Chọn một đối tượng"}
        </span>
        <div className="w-px h-4 bg-gray-700" />
        <button onClick={bringToFront} className="px-2 py-1 rounded text-xs text-gray-300 hover:bg-gray-700 transition">↑↑ Top</button>
        <button onClick={bringForward} className="px-2 py-1 rounded text-xs text-gray-300 hover:bg-gray-700 transition">↑ Lên</button>
        <button onClick={sendBackward} className="px-2 py-1 rounded text-xs text-gray-300 hover:bg-gray-700 transition">↓ Xuống</button>
        <div className="w-px h-4 bg-gray-700" />
        <button onClick={exportCanvas} className="px-3 py-1 rounded-lg text-xs bg-indigo-600 hover:bg-indigo-500 text-white transition">
          ⬇ Export PNG
        </button>
      </div>

      {/* Canvas */}
      <div
        className="relative rounded-xl overflow-hidden"
        style={{
          width: canvasW,
          height: canvasH,
          boxShadow: "0 0 0 1px rgba(255,255,255,0.08)",
        }}
      >
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10 rounded-xl">
            <div className="flex flex-col items-center gap-2">
              <span className="animate-spin w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full" />
              <span className="text-gray-400 text-xs">Đang render canvas...</span>
            </div>
          </div>
        )}
        <canvas ref={canvasRef} />
      </div>

      <p className="text-gray-600 text-xs">
        {canvasW} × {canvasH}px · scale {scale.toFixed(3)}x · gốc {origW} × {origH}px
      </p>
    </div>
  );
}