export default function UploadButton({ onSelect, isUploading }) {
  return (
    <label className="w-40 h-40 border-2 border-dashed rounded-xl flex items-center justify-center cursor-pointer hover:bg-blue-900 transition">
      <input
        type="file"
        accept="image/*"
        onChange={e => e.target.files[0] && onSelect(e.target.files[0])}
        className="hidden"
      />
      {isUploading ? (
        <div className="text-blue-300 text-sm">上传中...</div>
      ) : (
        <svg
          className="w-10 h-10 text-blue-300"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
        </svg>
      )}
    </label>
  );
}
