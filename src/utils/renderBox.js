import cv from "@techstark/opencv-js";

export class Colors {
  constructor() {
    this.palette = ["#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A"];
    this.n = this.palette.length;
  }
  get = (i) => this.palette[Math.floor(i) % this.n];
  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? `rgba(${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}, ${alpha})` : null;
  };
}

/**
 * Render Polygons and Boxes
 * @param {HTMLCanvasElement} canvas
 * @param {Array} boxes_data
 * @param {Array} labels
 * @param {Object} masks_protos (The output1 tensor from ONNX)
 */
export const renderBoxes = (canvas, boxes_data, labels, masks_protos) => {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const colors = new Colors();
  const font = "18px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";

  // --- PREPARE MASKS (YOLO Segmentation Logic) ---
  if (masks_protos) {
    const proto_h = 160;
    const proto_w = 160;
    const proto_dim = 32;
    const proto_flat = masks_protos.data; // Float32Array

    boxes_data.forEach((box) => {
      // 1. Generate 160x160 Mask for this object
      const maskMat = new cv.Mat(proto_h, proto_w, cv.CV_32FC1);
      
      // Matmul: Mask Weights * Protos
      for (let y = 0; y < proto_h; y++) {
          for (let x = 0; x < proto_w; x++) {
              let sum = 0;
              for (let i = 0; i < proto_dim; i++) {
                  sum += box.maskWeights[i] * proto_flat[i * proto_h * proto_w + y * proto_w + x];
              }
              const val = 1 / (1 + Math.exp(-sum)); // Sigmoid
              maskMat.floatPtr(y, x)[0] = val;
          }
      }

      const scaleX = canvas.width / proto_w;
      const scaleY = canvas.height / proto_h;

      const pad = 2; 
      const mx = Math.max(0, Math.floor(box.x / scaleX) - pad);
      const my = Math.max(0, Math.floor(box.y / scaleY) - pad);
      const mw = Math.min(proto_w - mx, Math.ceil(box.w / scaleX) + 2 * pad);
      const mh = Math.min(proto_h - my, Math.ceil(box.h / scaleY) + 2 * pad);

      if (mw > 0 && mh > 0) {
          const roiRect = new cv.Rect(mx, my, mw, mh);
          const roiMat = maskMat.roi(roiRect);
          
          const target_w = Math.round(mw * scaleX);
          const target_h = Math.round(mh * scaleY);

          const resizedMask = new cv.Mat();
          cv.resize(roiMat, resizedMask, new cv.Size(target_w, target_h), 0, 0, cv.INTER_LINEAR);
          
          // Thresholding
          const binaryMask = new cv.Mat();
          cv.threshold(resizedMask, binaryMask, 0.5, 255, cv.THRESH_BINARY);
          binaryMask.convertTo(binaryMask, cv.CV_8UC1);

          // Find Contours
          const contours = new cv.MatVector();
          const hierarchy = new cv.Mat();
          cv.findContours(binaryMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

          // Draw Logic
          const colorHex = colors.get(box.classId);
          ctx.fillStyle = Colors.hexToRgba(colorHex, 0.4); 
          ctx.strokeStyle = colorHex;
          ctx.lineWidth = 2;

          for (let i = 0; i < contours.size(); ++i) {
              const contour = contours.get(i);
              const data = contour.data32S; 

              ctx.beginPath();
              
              const startX = mx * scaleX;
              const startY = my * scaleY;

              ctx.moveTo(data[0] + startX, data[1] + startY);
              for (let j = 2; j < data.length; j += 2) {
                  ctx.lineTo(data[j] + startX, data[j + 1] + startY);
              }
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
          }

          // Cleanup memory
          roiMat.delete();
          resizedMask.delete();
          binaryMask.delete();
          contours.delete();
          hierarchy.delete();
      }
      maskMat.delete();
    });
  }

  // --- DRAW BOXES & TEXT ---
  boxes_data.forEach((box) => {
    const klass = labels[box.classId];
    const score = (box.score * 100).toFixed(1);
    const color = colors.get(box.classId);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.w, box.h);

    ctx.fillStyle = color;
    const text = `${klass} ${score}%`;
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(box.x, box.y - 18, textWidth + 4, 18);
    
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(text, box.x + 2, box.y - 18);
  });
};