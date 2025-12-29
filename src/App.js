import React, { useState, useRef, useEffect } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const [inputMode, setInputMode] = useState("upload");

  // Refs
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const webcamRef = useRef(null);
  const urlInputRef = useRef(null);

  const modelName = "seablocker_model.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  useEffect(() => {
    cv["onRuntimeInitialized"] = async () => {
      const baseModelURL = `${process.env.PUBLIC_URL}/model`;

      try {
        setLoading({ text: `Loading ${modelName}...`, progress: 0 });

        const yolov11 = await InferenceSession.create(
          `${baseModelURL}/${modelName}`,
          { executionProviders: ['wasm'] }
        );

        setLoading({ text: "Warming up model...", progress: null });
        const tensor = new Tensor(
          "float32",
          new Float32Array(modelInputShape.reduce((a, b) => a * b)),
          modelInputShape
        );
        await yolov11.run({ images: tensor });

        setSession(yolov11);
        setLoading(null);
      } catch (e) {
        alert("Gagal memuat model! Pastikan file .onnx ada di folder public/model/");
        console.error(e);
        setLoading(null);
      }
    };
  }, []);

  const runWebcam = async () => {
    if (inputMode === 'webcam' && webcamRef.current && canvasRef.current && session) {
      if (webcamRef.current.readyState === 4) { // Video ready
        await detectImage(
          webcamRef.current, // Video Element
          canvasRef.current,
          session,
          topk,
          iouThreshold,
          scoreThreshold,
          modelInputShape
        );
      }
      requestAnimationFrame(runWebcam); // Loop
    }
  };

  useEffect(() => {
    if (inputMode === 'webcam') {
      runWebcam();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputMode, session]);

  return (
    <div className="App">
      {loading && (
        <Loader progress={loading.progress}>
          {loading.text}
        </Loader>
      )}

      <div className="header">
        <h1>YOLO11 Segmentation - SeaBlocker</h1>
        <div className="btn-container">
          <button onClick={() => setInputMode("upload")}>📁 Upload</button>
          <button onClick={() => setInputMode("url")}>🔗 URL</button>
          <button onClick={() => setInputMode("webcam")}>📷 Webcam</button>
        </div>
      </div>

      <div className="content">
        {/* VIEW 1: IMAGE / URL */}
        {(inputMode === "upload" || inputMode === "url") && (
          <>
            <img
              ref={imageRef}
              src={image || "#"}
              alt=""
              crossOrigin="anonymous"
              style={{ display: image ? "block" : "none", maxWidth: "100%" }}
              onLoad={() => {
                detectImage(
                  imageRef.current,
                  canvasRef.current,
                  session,
                  topk,
                  iouThreshold,
                  scoreThreshold,
                  modelInputShape
                );
              }}
            />
            <canvas
              id="canvas"
              width={modelInputShape[2]}
              height={modelInputShape[3]}
              ref={canvasRef}
              className="canvas-overlay"
            />
          </>
        )}

        {/* VIEW 2: WEBCAM */}
        {inputMode === "webcam" && (
          <div className="webcam-container">
            <video
              ref={webcamRef}
              autoPlay
              muted
              playsInline
              onLoadedMetadata={() => {
                webcamRef.current.width = 640;
                webcamRef.current.height = 640;
              }}
              style={{ position: "absolute", zIndex: 1 }}
            />
            <canvas
              ref={canvasRef}
              width={640}
              height={640}
              style={{ position: "absolute", zIndex: 2 }}
            />
          </div>
        )}
      </div>

      {/* INPUT CONTROLS */}
      {inputMode === "upload" && (
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (!e.target.files || e.target.files.length === 0) {
              return;
            }

            if (image) {
              URL.revokeObjectURL(image);
              setImage(null);
            }

            const url = URL.createObjectURL(e.target.files[0]);
            imageRef.current.src = url;
            setImage(url);
          }}
        />
      )}

      {inputMode === "url" && (
        <div className="url-input">
          <input ref={urlInputRef} type="text" placeholder="Paste Image URL here..." />
          <button onClick={() => setImage(urlInputRef.current.value)}>Detect</button>
        </div>
      )}

      {inputMode === "webcam" && (
        <button onClick={() => {
          navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
            .then(stream => {
              webcamRef.current.srcObject = stream;
            });
        }}>Start Camera</button>
      )}
    </div>
  );
};

export default App;