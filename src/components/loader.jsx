import React from "react";
import "../style/loader.css";

const Loader = ({ children, progress }) => {
  return (
    <div className="loader-overlay">
      <div className="loader-box">
        {/* Spinner Modern */}
        <div className="spinner"></div>

        {/* Text Status */}
        <p className="loader-text">{children}</p>

        {/* Progress Bar (Hanya muncul jika ada progress download) */}
        {progress !== null && progress !== undefined && (
          <div className="progress-container">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            ></div>
            <span className="progress-number">{progress}%</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default Loader;