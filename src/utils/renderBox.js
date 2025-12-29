import cv from "@techstark/opencv-js";

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
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const colors = new Colors();
  const font = "18px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";

  // --- PREPARE MASKS (YOLO Segmentation Logic) ---
  // output1 shape is [1, 32, 160, 160]
  // We need to matrix multiply: maskWeights [32] x Protos [32, 160, 160]
  
  if (masks_protos) {
    const proto_h = 160;
    const proto_w = 160;
    const proto_dim = 32;
    const proto_flat = masks_protos.data; // Float32Array

    boxes_data.forEach((box) => {
      // 1. Create a 160x160 mask for this specific object
      const maskMat = new cv.Mat(proto_h, proto_w, cv.CV_32FC1);
      
      // Perform manual Matrix Multiplication (JS is faster than looping OpenCV Mats for this specific shape)
      for (let y = 0; y < proto_h; y++) {
          for (let x = 0; x < proto_w; x++) {
              let sum = 0;
              for (let i = 0; i < proto_dim; i++) {
                  // box.maskWeights[i] * Proto[i][y][x]
                  sum += box.maskWeights[i] * proto_flat[i * proto_h * proto_w + y * proto_w + x];
              }
              // Sigmoid
              const val = 1 / (1 + Math.exp(-sum));
              maskMat.floatPtr(y, x)[0] = val;
          }
      }

      // 2. Resize mask to Box Size (Optimization: Process only ROI)
      // Map box coordinates from Canvas (e.g., 1920x1080) to Proto (160x160)
      const scaleX = proto_w / canvas.width;
      const scaleY = proto_h / canvas.height;
      
      const mx = Math.max(0, Math.floor(box.x * scaleX));
      const my = Math.max(0, Math.floor(box.y * scaleY));
      const mw = Math.min(proto_w - mx, Math.ceil(box.w * scaleX));
      const mh = Math.min(proto_h - my, Math.ceil(box.h * scaleY));

      if (mw > 0 && mh > 0) {
          // Crop the ROI from the 160x160 mask
          const roiRect = new cv.Rect(mx, my, mw, mh);
          const roiMat = maskMat.roi(roiRect);
          
          // Resize ROI to actual Bounding Box size
          const resizedMask = new cv.Mat();
          cv.resize(roiMat, resizedMask, new cv.Size(box.w, box.h), 0, 0, cv.INTER_LINEAR);
          
          // Threshold to make it binary (0 or 255)
          const binaryMask = new cv.Mat();
          cv.threshold(resizedMask, binaryMask, 0.5, 255, cv.THRESH_BINARY);
          binaryMask.convertTo(binaryMask, cv.CV_8UC1);

          // 3. Find Contours (The Polygon!)
          const contours = new cv.MatVector();
          const hierarchy = new cv.Mat();
          cv.findContours(binaryMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

          // 4. Draw Contours on Canvas
          const colorHex = colors.get(box.classId);
          ctx.fillStyle = Colors.hexToRgba(colorHex, 0.4); // Semi-transparent fill
          ctx.strokeStyle = colorHex;
          ctx.lineWidth = 2;

          for (let i = 0; i < contours.size(); ++i) {
              const contour = contours.get(i);
              const data = contour.data32S; // Array of [x, y, x, y...]

              ctx.beginPath();
              // Offset by box position (since we processed relative to box)
              ctx.moveTo(data[0] + box.x, data[1] + box.y);
              for (let j = 2; j < data.length; j += 2) {
                  ctx.lineTo(data[j] + box.x, data[j + 1] + box.y);
              }
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
          }

          // Cleanup memory per box
          roiMat.delete();
          resizedMask.delete();
          binaryMask.delete();
          contours.delete();
          hierarchy.delete();
      }
      maskMat.delete();
    });
  }

  // --- DRAW BOXES & TEXT (Layered on top of polygons) ---
  boxes_data.forEach((box) => {
    const klass = labels[box.classId];
    const score = (box.score * 100).toFixed(1);
    const color = colors.get(box.classId);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.w, box.h);

    // Label
    ctx.fillStyle = color;
    const text = `${klass} ${score}%`;
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(box.x, box.y - 18, textWidth + 4, 18);
    
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(text, box.x + 2, box.y - 18);
  });
};

// Helper Colors class (put this at the bottom or import it)
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