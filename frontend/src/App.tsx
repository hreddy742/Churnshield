import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import PredictPage from "./pages/PredictPage";
import ModelPage from "./pages/ModelPage";
import MonitorPage from "./pages/MonitorPage";

function Nav() {
  const base = "px-4 py-2 text-sm font-medium rounded-lg transition-colors";
  const active = `${base} bg-blue-50 text-blue-700`;
  const inactive = `${base} text-gray-600 hover:bg-gray-100`;

  return (
    <header className="sticky top-0 z-10 bg-white border-b border-gray-200">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-gray-900">ChurnShield</span>
          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">
            open source
          </span>
        </div>
        <nav className="flex items-center gap-1">
          <NavLink to="/" end className={({ isActive }) => isActive ? active : inactive}>
            Predict
          </NavLink>
          <NavLink to="/model" className={({ isActive }) => isActive ? active : inactive}>
            Model
          </NavLink>
          <NavLink to="/monitor" className={({ isActive }) => isActive ? active : inactive}>
            Monitor
          </NavLink>
        </nav>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Nav />
        <main>
          <Routes>
            <Route path="/" element={<PredictPage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/monitor" element={<MonitorPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
