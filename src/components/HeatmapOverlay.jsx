export default function HeatmapOverlay({ image, heatmap, opacity = 0.5 }) {
  return (
    <div className="relative w-full overflow-hidden rounded border border-gray-200 dark:border-gray-700">
      <img src={image} className="w-full block" alt="mri" />
      {heatmap && (
        <img
          src={heatmap}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          style={{
            opacity,
            mixBlendMode: "multiply",
            filter: `contrast(120%) brightness(110%)`,
          }}
          alt="heatmap"
        />
      )}
    </div>
  );
}