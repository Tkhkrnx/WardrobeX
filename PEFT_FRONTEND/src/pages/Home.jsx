import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadButton from '../components/UploadButton';
import CameraButton from '../components/CameraButton';

export default function Home() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [progress, setProgress] = useState(0);
  const [modalImg, setModalImg] = useState(null);
  const navigate = useNavigate();

  const handleFileSelect = (newFile) => {
    setFile(null);
    setFileName('');
    setProgress(0);

    const reader = new FileReader();

    reader.onloadstart = () => {
      setProgress(10);
    };

    reader.onload = () => {
      setProgress(100);
      setFile(reader.result);
      setFileName(newFile.name);
      sessionStorage.setItem('uploaded', reader.result);
    };

    reader.onerror = () => {
      alert('图片读取失败');
      setProgress(0);
    };

    reader.readAsDataURL(newFile);
  };

  const toResult = () => {
    if (!file || progress < 100) return alert('请等待图片上传完成');
    navigate('/result', { state: { file, fileName } });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4">
      <h1 className="text-4xl font-extrabold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-300 to-green-300 animate-pulse">
        WardrobeX
      </h1>

      {/* 新增帮助提示 */}
      <p className="max-w-md mb-8 text-center text-sm sm:text-base text-gray-300">
        请选择或拍摄一张衣物照片，等待上传完成后点击“生成穿搭推荐”，获得个性化穿搭建议。
      </p>

      <div className="flex flex-col sm:flex-row gap-6 mb-6">
        <UploadButton onSelect={handleFileSelect} isUploading={progress > 0 && progress < 100} />
        <CameraButton onSelect={handleFileSelect} isUploading={progress > 0 && progress < 100} />
      </div>

      {progress > 0 && (
        <div className="w-64 h-2 bg-white/20 rounded-full overflow-hidden mb-4">
          <div
            className="bg-green-400 h-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )}

      {file && progress >= 100 && (
        <div className="mb-4 text-green-300 text-sm flex flex-col items-center">
          <span>✅ 已上传：{fileName || '照片'}</span>
          <img
            src={file}
            alt="上传预览"
            className="mt-2 w-40 h-40 object-contain rounded-md cursor-pointer"
            onClick={() => setModalImg(file)}
          />
        </div>
      )}

      <button
        onClick={toResult}
        disabled={!file || progress < 100}
        className={`px-8 py-3 rounded-full shadow-lg transition ${
          file && progress >= 100
            ? 'bg-blue-500 hover:bg-blue-600 cursor-pointer'
            : 'bg-gray-600 cursor-not-allowed'
        }`}
      >
        生成穿搭推荐
      </button>

      {/* 大图弹窗 */}
      {modalImg && (
        <div
          className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 cursor-zoom-out"
          onClick={() => setModalImg(null)}
        >
          <img
            src={modalImg}
            alt="大图预览"
            className="max-w-[90vw] max-h-[90vh] rounded-md shadow-lg"
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={() => setModalImg(null)}
            className="absolute top-6 right-6 text-white text-3xl font-bold cursor-pointer bg-black bg-opacity-50 rounded-full px-3"
            aria-label="关闭"
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}
