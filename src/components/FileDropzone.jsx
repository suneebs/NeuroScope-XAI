import { useRef, useState } from "react";
import PropTypes from "prop-types";

export default function FileDropzone({ onFile, accept }) {
  const [active, setActive] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = (files) => {
    if (!files || !files[0]) return;
    onFile(files[0]);
  };

  return (
    <div
      className={[
        "border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition",
        active ? "border-brand bg-indigo-50/50 dark:bg-indigo-950/20" : "border-gray-300 dark:border-gray-700",
      ].join(" ")}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setActive(true);
      }}
      onDragLeave={() => setActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setActive(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={(e) => handleFiles(e.target.files)}
        className="hidden"
      />
      <div className="text-sm text-gray-600 dark:text-gray-300">
        Drop an image here, or click to browse
      </div>
      <div className="text-xs text-gray-500 mt-1">PNG • JPG • JPEG</div>
    </div>
  );
}

FileDropzone.propTypes = {
  onFile: PropTypes.func.isRequired,
  accept: PropTypes.string,
};