import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function Result() {
  const location = useLocation();
  const navigate = useNavigate();

  const fileBase64 = location.state?.file;
  const fileName = location.state?.fileName || 'photo.png';

  const [recommendation, setRecommendation] = useState('');
  const [displayText, setDisplayText] = useState('');
  const [loading, setLoading] = useState(false);

  // base64转blob函数
  function base64ToBlob(base64) {
    const arr = base64.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    return new Blob([u8arr], { type: mime });
  }

  useEffect(() => {
    if (!fileBase64) {
      alert('未检测到上传图片，请返回重新上传');
      navigate('/');
      return;
    }

    async function fetchRecommendation() {
      setLoading(true);
      setRecommendation('');
      try {
        const imgBlob = base64ToBlob(fileBase64);
        const formData = new FormData();
        formData.append('file', imgBlob, fileName);

        const res = await fetch('/recommend', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) throw new Error(`请求失败，状态码 ${res.status}`);

        const data = await res.json();

        if (!data.outfit_description) throw new Error('接口返回无推荐文案');

        setRecommendation(data.outfit_description);
      } catch (error) {
        alert('推荐请求失败：' + error.message);
        navigate('/');
      } finally {
        setLoading(false);
      }
    }

    fetchRecommendation();
  }, [fileBase64, fileName, navigate]);

  // 逐字动画
  useEffect(() => {
    if (!recommendation) {
      setDisplayText('');
      return;
    }
    let index = 0;
    setDisplayText('');
    const interval = setInterval(() => {
      setDisplayText(recommendation.substring(0, index + 1));
      index++;
      if (index >= recommendation.length) clearInterval(interval);
    }, 40);
    return () => clearInterval(interval);
  }, [recommendation]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-black via-blue-900 to-green-900 text-white">
      <h1 className="text-4xl font-extrabold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-300 to-green-300 animate-pulse">
        WardrobeX
      </h1>

      <div className="mb-6">
        {/* 点击图片查看大图功能 */}
        <img
          src={fileBase64}
          alt="上传图片"
          className="w-48 h-48 object-contain rounded-md cursor-pointer"
          onClick={() => window.open(fileBase64, '_blank')}
        />
      </div>

      <div
        className={`w-full max-w-md bg-white/10 p-6 rounded-lg shadow-inner transition-opacity duration-1000 ${
          loading ? 'opacity-50' : 'opacity-100'
        }`}
        style={{
          fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
          fontSize: '16px',
          color: '#D1D5DB',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          textAlign: 'left',
          lineHeight: 1.5,
          userSelect: 'text',
          // 关键：不要 maxHeight，去掉 overflowY，允许自动撑开
        }}
      >
        {loading ? <p className="text-green-300">生成中...</p> : displayText}
      </div>

      <button
        onClick={() => navigate('/')}
        className="mt-10 px-6 py-3 rounded-full bg-blue-500 hover:bg-blue-600 transition shadow-lg text-white text-sm sm:text-base"
      >
        返回首页
      </button>
    </div>
  );
}
