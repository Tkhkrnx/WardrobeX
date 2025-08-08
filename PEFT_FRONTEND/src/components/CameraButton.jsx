export default function CameraButton({ onSelect, isUploading }) {
  return (
    <label className="w-40 h-40 bg-blue-600 rounded-xl flex items-center justify-center cursor-pointer hover:bg-blue-500 transition">
      <input
        type="file"
        accept="image/*"
        capture="environment"
        onChange={e => e.target.files[0] && onSelect(e.target.files[0])}
        className="hidden"
      />
      {isUploading ? (
        <div className="text-white text-sm">上传中...</div>
      ) : (
        <svg
          className="w-10 h-10 text-white"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 7h4l2-3h6l2 3h4v12H3V7z"
          />
          <circle cx="12" cy="13" r="3" />
        </svg>
      )}
    </label>
  );
}
