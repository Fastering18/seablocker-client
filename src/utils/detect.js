import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox"; 
import labels from "./labels.json"; 

export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  if (!session) return;

  const [modelWidth, modelHeight] = inputShape.slice(2);

  const mat = cv.imread(image); 
  const matC3 = new cv.Mat();
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB); 
  
  const input = cv.blobFromImage(
    matC3,
    1 / 255.0, 
    new cv.Size(modelWidth, modelHeight),
    new cv.Scalar(0, 0, 0),
    true, 
    false 
  );

  const tensor = new Tensor("float32", input.data32F, inputShape); 

  // Inference
  const config = { images: tensor };
  const { output0, output1 } = await session.run(config); 

  const boxes = [];
  
  // --- PARSING OUTPUT ---
  const numAnchors = output0.dims[2]; 
  // const numOut = output0.dims[1];     
  const numClasses = 3;               
  const data = output0.data;

  for (let i = 0; i < numAnchors; i++) {
    let maxScore = 0;
    let maxClass = -1;
    
    for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numAnchors + i];
        if (score > maxScore) {
            maxScore = score;
            maxClass = c;
        }
    }

    if (maxScore > scoreThreshold) {
        const cx = data[0 * numAnchors + i];
        const cy = data[1 * numAnchors + i];
        const w = data[2 * numAnchors + i];
        const h = data[3 * numAnchors + i];

        const x = (cx - w / 2) * (image.width / modelWidth); 
        const y = (cy - h / 2) * (image.height / modelHeight);
        const width = w * (image.width / modelWidth);
        const height = h * (image.height / modelHeight);

        const weights = [];
        for (let m = 0; m < 32; m++) {
            weights.push(data[(4 + numClasses + m) * numAnchors + i]);
        }

        boxes.push({
            x: x, y: y, w: width, h: height,
            score: maxScore,
            classId: maxClass,
            maskWeights: weights 
        });
    }
  }

  // --- NMS ---
  boxes.sort((a, b) => b.score - a.score);
  const result = [];
  while (boxes.length > 0) {
      const best = boxes.shift();
      result.push(best);
      for (let i = boxes.length - 1; i >= 0; i--) {
          if (iou(best, boxes[i]) > iouThreshold) {
              boxes.splice(i, 1);
          }
      }
  }

  if (result.length > 0) {
      console.log(`Terdeteksi: ${result.length} objek`);
      
      renderBoxes(canvas, result, labels, output1); 
  }

  mat.delete();
  matC3.delete();
  input.delete();
};

function iou(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.w, box2.x + box2.w);
    const y2 = Math.min(box1.y + box1.h, box2.y + box2.h);
    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const box1Area = box1.w * box1.h;
    const box2Area = box2.w * box2.h;
    return interArea / (box1Area + box2Area - interArea);
}