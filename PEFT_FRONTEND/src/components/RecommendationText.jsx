export default function RecommendationText({ text }) {
  return (
    <div className="mt-6 w-full max-w-md bg-white/10 backdrop-blur-lg p-4 rounded-xl text-white">
      <h2 className="text-lg font-semibold mb-2">推荐文案：</h2>
      <div className="typewriter text-sm">{text}</div>
    </div>
  );
}
