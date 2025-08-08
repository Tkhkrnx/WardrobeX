// App.jsx
import { Routes, Route } from 'react-router-dom';
import HomePage from './pages/Home';
import Result from './pages/Result';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/result" element={<Result />} />
    </Routes>
  );
}
